from __future__ import annotations

import difflib
import hmac
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Receive, Scope, Send

from rag_assistant import __version__
from rag_assistant.auth import create_token, verify_token
from rag_assistant.config import AppConfig
from rag_assistant.document_processor import DocumentProcessingError, parse_document
from rag_assistant.evals import run_retrieval_eval
from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.llm import AnswerGenerator, LLMError
from rag_assistant.metadata import create_metadata_store
from rag_assistant.models import (
    BulkIngestRequest,
    BulkIngestResponse,
    Citation,
    ContextChunk,
    DeleteResponse,
    DocumentsResponse,
    DocumentSummary,
    ErrorResponse,
    EvalRequest,
    EvalResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    FileIngestResponse,
    IngestAsyncResponse,
    IngestJobStatus,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RouteMetrics,
    SemanticQueryRequest,
    SessionClearResponse,
    SessionsListResponse,
    TokenRequest,
    TokenResponse,
    WorkspaceInfo,
    WorkspacesResponse,
)
from rag_assistant.stores import create_vector_store

app_config = AppConfig.from_env()

if app_config.environment == "production" and not app_config.api_key:
    raise RuntimeError(
        "RAG_ENV=production requires RAG_API_KEY to be set. "
        "Refusing to start with authentication disabled."
    )

answer_generator = AnswerGenerator(
    provider=app_config.llm_provider,
    model=app_config.llm_model,
    api_key=app_config.llm_api_key,
    base_url=app_config.llm_base_url,
    timeout_seconds=app_config.llm_timeout_seconds,
    max_output_tokens=app_config.llm_max_output_tokens,
)

kb = KnowledgeBase(
    chunk_size=app_config.default_chunk_size,
    chunk_overlap=app_config.default_chunk_overlap,
    storage_path=app_config.storage_path,
    default_embedding_model=app_config.default_embedding_model,
    vector_store=create_vector_store(
        app_config.vector_store,
        app_config.storage_path,
        qdrant_url=app_config.qdrant_url,
        qdrant_api_key=app_config.qdrant_api_key,
        qdrant_collection=app_config.qdrant_collection,
    ),
    metadata_store=create_metadata_store(storage_path=app_config.storage_path, db_url=app_config.db_url),
    reranker=app_config.reranker,
    reranker_model=app_config.reranker_model,
    query_rewriter=answer_generator.rewrite_query,
)

logging.basicConfig(level=getattr(logging, app_config.log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag-knowledge-assistant")

if not app_config.api_key:
    logger.warning(
        "RAG_API_KEY is not set; authentication is DISABLED. "
        "Set RAG_API_KEY before exposing this service beyond localhost."
    )

app = FastAPI(
    title="TrustAware RAG Knowledge Assistant",
    version=__version__,
    description=("Document-grounded RAG API with hybrid retrieval, reranking, citations, "
                 "confidence-aware no-answer handling, and knowledge diagnostics."),
    docs_url="/docs" if (app_config.docs_enabled and not app_config.api_key) else None,
    redoc_url="/redoc" if (app_config.docs_enabled and not app_config.api_key) else None,
    openapi_url="/openapi.json" if (app_config.docs_enabled and not app_config.api_key) else None,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UI_CANDIDATES = (
    _PROJECT_ROOT / "ui",
    Path.cwd() / "ui",
    Path.cwd() / "rag-knowledge-assistant-main" / "ui",
)
_UI_DIR = next((candidate.resolve() for candidate in _UI_CANDIDATES if candidate.is_dir()),
               _PROJECT_ROOT / "ui")
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

if app_config.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_config.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


class _BodyLimitExceeded(Exception):
    """Raised while streaming a request body that exceeds the configured byte limit."""


class _BodyLimitMiddleware:
    """Pure-ASGI middleware enforcing a streamed request-body byte limit.

    The HTTP middleware below only inspects the ``Content-Length`` header, which
    chunked-transfer requests do not carry. This middleware counts the actual
    bytes as they arrive so no request can bypass the limit, and signals
    over-limit bodies with :class:`_BodyLimitExceeded`, which the HTTP middleware
    converts into a structured 413 response.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return
        seen = 0

        async def guarded_receive() -> dict:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyLimitExceeded()
            return message

        await self.app(scope, guarded_receive, send)


# Registered before the @app.middleware("http") decorator below so the HTTP
# middleware sits outside it and can translate _BodyLimitExceeded into a 413.
app.add_middleware(_BodyLimitMiddleware, max_bytes=app_config.request_body_limit_bytes)


def _is_body_limit_exceeded(exc: BaseException) -> bool:
    """Walk the exception cause chain looking for _BodyLimitExceeded.

    Frameworks such as FastAPI wrap body-read failures (they turn the raw
    receive error into a generic 400), so the signal can be several layers
    deep when the exception reaches the middleware.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, _BodyLimitExceeded):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False

_START_TIME = time.time()


@dataclass
class _RouteMetric:
    requests: int = 0
    errors: int = 0
    total_duration_ms: float = 0.0
    last_status: int | None = None

    @property
    def avg_response_ms(self) -> float:
        return round(self.total_duration_ms / self.requests, 3) if self.requests else 0.0


class _MetricsCollector:
    def __init__(self) -> None:
        self._metrics: dict[str, _RouteMetric] = {}
        self._requests = 0
        self._errors = 0
        self._lock = Lock()

    def record(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        key = f"{method} {path}"
        with self._lock:
            metric = self._metrics.setdefault(key, _RouteMetric())
            metric.requests += 1
            metric.total_duration_ms += duration_ms
            metric.last_status = status_code
            self._requests += 1
            if status_code >= 400:
                metric.errors += 1
                self._errors += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            return MetricsResponse(
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=round(time.time() - _START_TIME, 2),
                requests_total=self._requests,
                errors_total=self._errors,
                routes={k: RouteMetrics(requests=v.requests, errors=v.errors,
                                        avg_response_ms=v.avg_response_ms, last_status=v.last_status)
                        for k, v in self._metrics.items()},
            )


class _RateLimiter:
    def __init__(self, requests: int, window_seconds: int) -> None:
        self._requests = requests
        self._window_seconds = max(1, window_seconds)
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int, int]:
        if self._requests <= 0:
            return True, 0, 0
        now = time.time()
        cutoff = now - self._window_seconds
        with self._lock:
            history = self._timestamps[key]
            while history and history[0] <= cutoff:
                history.popleft()
            if len(history) >= self._requests:
                reset = max(1, int(history[0] + self._window_seconds - now))
                return False, 0, reset
            history.append(now)
            if len(self._timestamps) > 5000:
                stale_keys = [k for k, values in self._timestamps.items() if not values or values[-1] <= cutoff]
                for stale_key in stale_keys[:1000]:
                    self._timestamps.pop(stale_key, None)
            return True, self._requests - len(history), self._window_seconds


_metrics = _MetricsCollector()
_limiter = _RateLimiter(app_config.rate_limit_requests, app_config.rate_limit_window_seconds)


def _error_code(status_code: int) -> str:
    return {
        400: "invalid_request", 401: "unauthorized", 404: "not_found", 413: "payload_too_large",
        422: "validation_error", 429: "rate_limit_exceeded", 503: "dependency_failure",
    }.get(status_code, "internal_error" if status_code >= 500 else "request_failed")


def _error_payload(request: Request, status_code: int, message: str, details: dict | None = None) -> ErrorResponse:
    return ErrorResponse(error_code=_error_code(status_code), message=message,
                         request_id=getattr(request.state, "request_id", None),
                         path=request.url.path, details=details)


def _error_response(request: Request, status_code: int, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=_error_payload(request, status_code, message, details).model_dump())


def _apply_response_headers(response: JSONResponse, request: Request, path: str) -> JSONResponse:
    response.headers["x-request-id"] = request.state.request_id
    if not _rate_limit_exempt(path):
        response.headers["x-ratelimit-limit"] = str(app_config.rate_limit_requests)
        response.headers["x-ratelimit-remaining"] = str(getattr(request.state, "rate_limit_remaining", 0))
        response.headers["x-ratelimit-reset-after"] = str(getattr(request.state, "rate_limit_reset_after", 0))
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["x-frame-options"] = "DENY"
    # The UI is served inline (Phase 1 UI policy); deny everything else.
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'"
    )
    return response


def _client_key(request: Request) -> str:
    """Rate-limit identity (Phase 4.2): workspace/user claim when authed, else the
    API key (hashed prefix, shared for the bootstrap admin), else client IP."""
    workspace_id = getattr(request.state, "workspace_id", None)
    if workspace_id:
        return f"workspace:{workspace_id}"
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    api_key = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
    if api_key:
        import hashlib

        return f"key:{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]}"
    if app_config.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client and request.client.host else "anonymous"


def _rate_limit_exempt(path: str) -> bool:
    return path in {"/health", "/healthz", "/ready", "/metrics"} or path.startswith(("/docs", "/redoc")) or path == "/openapi.json"


@app.middleware("http")
async def middleware(request: Request, call_next):
    started = time.perf_counter()
    request.state.request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    path = request.url.path

    if not _rate_limit_exempt(path) and app_config.rate_limit_requests > 0:
        allowed, remaining, reset_after = _limiter.allow(_client_key(request))
        request.state.rate_limit_remaining = remaining if allowed else 0
        request.state.rate_limit_reset_after = reset_after
        if not allowed:
            response = _error_response(request, 429, "Rate limit exceeded. Retry later.")
            response.headers["Retry-After"] = str(reset_after)
            _metrics.record(request.method, path, 429, (time.perf_counter() - started) * 1000)
            return _apply_response_headers(response, request, path)

    # Early rejection for oversized or malformed declared bodies. Actual streamed
    # bytes are enforced by _BodyLimitMiddleware (chunked requests carry no
    # Content-Length header at all). Responses are returned directly: raising
    # HTTPException from middleware bypasses the structured exception handlers.
    content_length = request.headers.get("content-length")
    if content_length and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        try:
            declared_length = int(content_length)
        except ValueError:
            response = _error_response(request, 400, "Invalid Content-Length header.")
            _metrics.record(request.method, path, 400, (time.perf_counter() - started) * 1000)
            return _apply_response_headers(response, request, path)
        if declared_length > app_config.request_body_limit_bytes:
            response = _error_response(request, 413, "Request body exceeds the configured limit.")
            _metrics.record(request.method, path, 413, (time.perf_counter() - started) * 1000)
            return _apply_response_headers(response, request, path)

    try:
        response = await call_next(request)
    except _BodyLimitExceeded:
        response = _error_response(request, 413, "Request body exceeds the configured limit.")
        _metrics.record(request.method, path, 413, (time.perf_counter() - started) * 1000)
        return _apply_response_headers(response, request, path)
    except HTTPException as exc:
        if _is_body_limit_exceeded(exc):
            response = _error_response(request, 413, "Request body exceeds the configured limit.")
            _metrics.record(request.method, path, 413, (time.perf_counter() - started) * 1000)
            return _apply_response_headers(response, request, path)
        _metrics.record(request.method, path, exc.status_code, (time.perf_counter() - started) * 1000)
        raise
    except Exception:
        _metrics.record(request.method, path, 500, (time.perf_counter() - started) * 1000)
        logger.exception("Unhandled request failure: %s %s", request.method, path)
        raise

    _metrics.record(request.method, path, response.status_code, (time.perf_counter() - started) * 1000)
    return _apply_response_headers(response, request, path)


async def _guard_api_key(request: Request, x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    """AuthN/Z (Phase 4.1): static API key (bootstrap admin) or a JWT with a workspace claim.

    The caller's workspace lands on ``request.state.workspace_id``; routes pass it
    to the KB, which enforces it as a server-side metadata filter on every
    operation (query/ingest/delete/list).
    """
    if app_config.api_key and hmac.compare_digest((x_api_key or "").encode("utf-8"),
                                                  app_config.api_key.encode("utf-8")):
        request.state.user_id = "admin"
        request.state.workspace_id = None
        return
    if app_config.jwt_secret and x_api_key:
        claims = verify_token(x_api_key, app_config)
        if claims is not None and claims.get("workspace_id"):
            request.state.user_id = claims.get("sub")
            request.state.workspace_id = claims.get("workspace_id")
            return
    if not app_config.api_key:
        return  # development mode: authentication disabled
    raise HTTPException(status_code=401, detail="Missing or invalid API key.", headers={"WWW-Authenticate": "ApiKey"})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    payload = _error_payload(request, exc.status_code, str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    # FastAPI wraps body-read failures (including our streamed body-limit
    # rejection) in a starlette.HTTPException; recover the 413 contract there.
    if _is_body_limit_exceeded(exc):
        return _error_response(request, 413, "Request body exceeds the configured limit.")
    # Preserve the FastAPI/Starlette default shape for all other errors.
    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = _error_payload(request, 422, "Request validation failed.", {"errors": exc.errors()})
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error: %s", exc)
    payload = _error_payload(request, 500, "An unexpected error occurred while processing the request.")
    return JSONResponse(status_code=500, content=payload.model_dump())


def _source_citations(chunks: list) -> list[Citation]:
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks, start=1):
        page_match = re.findall(r"\[PAGE\s+(\d+)\]", chunk.text, flags=re.IGNORECASE)
        page = page_match[-1] if page_match else chunk.metadata.get("page")
        label = str(chunk.metadata.get("filename") or chunk.source_id)
        if page:
            label += f" — Page {page}"
        else:
            label += f" — Chunk {chunk.chunk_index}"
        citations.append(Citation(citation_id=index, source_id=chunk.source_id,
                                  chunk_index=chunk.chunk_index, label=label, metadata=chunk.metadata))
    return citations


def _confidence(chunks: list, retrieval: str) -> float:
    if not chunks:
        return 0.0
    best = max(float(chunk.score) for chunk in chunks)
    if retrieval == "hybrid":
        # RRF scores are small; relative separation is more meaningful than the raw score.
        if len(chunks) == 1:
            return 0.72
        second = float(chunks[1].score)
        separation = max(0.0, (best - second) / max(best, 1e-9))
        return round(min(0.92, 0.25 + min(0.45, best * 12.0) + 0.2 * separation), 3)
    return round(min(0.98, max(0.0, best)), 3)


def _parse_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _diagnose_sources(chunks: list) -> tuple[list[str], bool]:
    warnings: list[str] = []
    if not chunks:
        return ["No sufficiently relevant evidence was retrieved."], True

    unique_sources = {}
    for chunk in chunks:
        unique_sources.setdefault(chunk.source_id, chunk)
    dated = [(source_id, _parse_date(c.metadata.get("updated_at") or c.metadata.get("document_date") or c.metadata.get("ingested_at")))
             for source_id, c in unique_sources.items()]
    dated = [(sid, dt) for sid, dt in dated if dt is not None]
    if len(dated) >= 2:
        newest = max(dt for _, dt in dated)
        oldest = min(dt for _, dt in dated)
        if (newest - oldest).days >= 180:
            warnings.append("Potentially outdated information detected: sources have substantially different update dates.")

    numeric_patterns = re.compile(r"\b\d+(?:\.\d+)?\b")
    signatures = []
    for source_id, chunk in unique_sources.items():
        nums = numeric_patterns.findall(chunk.text)
        terms = set(re.findall(r"[a-zA-Z]{4,}", chunk.text.lower()))
        signatures.append((source_id, set(nums), terms))
    for i, (_source_a, nums_a, terms_a) in enumerate(signatures):
        for _source_b, nums_b, terms_b in signatures[i + 1:]:
            if nums_a and nums_b and nums_a != nums_b and len(terms_a & terms_b) >= 4:
                warnings.append("Potential conflicting information detected across retrieved sources. Review the cited documents.")
                return warnings, False
    return warnings, False


def _dedup_context(chunks: list) -> list:
    """Collapse near-duplicate context chunks (SequenceMatcher > 0.85) — keeps order."""
    if len(chunks) < 2:
        return chunks
    kept: list = []
    seen: list[str] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk.text).strip()
        if any(difflib.SequenceMatcher(None, text, prev).ratio() > 0.85 for prev in seen):
            continue
        seen.append(text)
        kept.append(chunk)
    return kept


def _trim_context(chunks: list, max_tokens: int) -> list:
    """Trim context to a token budget (0 = unlimited); truncates the tail."""
    if not max_tokens or max_tokens <= 0 or not chunks:
        return chunks
    kept: list = []
    used = 0
    for chunk in chunks:
        tokens = max(1, len(chunk.text) // 4)
        if used + tokens > max_tokens:
            break
        used += tokens
        kept.append(chunk)
    return kept


def _strip_page_markers(text: str) -> str:
    """Strip `[PAGE n]` markers from context text (page lives in metadata)."""
    return re.sub(r"\s*\[PAGE\s+\d+\]\s*", " ", text).strip()


def _verify_citations(answer: str, citations: list[Citation], context: list[ContextChunk]) -> list[dict[str, Any]]:
    """Lexical citation verification (full NLI/judge verification is deferred).

    For every citation: was it referenced in the answer? When yes, does the
    answer share meaningful lexical ground with the cited chunk text?
    """
    cited_ids = {int(match.group(1)) for match in re.finditer(r"\[(\d+)\]", answer)}
    report: list[dict[str, Any]] = []
    for citation, chunk in zip(citations, context, strict=True):
        cited = citation.citation_id in cited_ids
        overlap = 0.0
        if cited and chunk.text:
            answer_terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
            if chunk_terms:
                overlap = len(answer_terms & chunk_terms) / len(chunk_terms)
        report.append({"citation_id": citation.citation_id, "cited": cited, "chunk_token_overlap": round(overlap, 3)})
    return report


def _grounded_status(answer: str, citations: list[Citation], context: list[ContextChunk], *,
                     used_llm: bool) -> tuple[bool, list[dict[str, Any]]]:
    """Compute ``grounded`` — extractive answers are grounded by construction;
    LLM answers are verified: every citation must exist and be supported by the
    answer's lexical ground overlap with the cited chunk (NLI/judge deferred)."""
    if not used_llm:
        return True, []
    verification = _verify_citations(answer, citations, context)
    grounded = bool(verification) and all(
        item["cited"] and item["chunk_token_overlap"] >= 0.05 for item in verification
    )
    return grounded, verification


def _build_response(question: str, chunks: list, trace: dict, payload: QueryRequest) -> QueryResponse:
    chunks = _trim_context(_dedup_context(chunks), payload.context_max_tokens)
    context = [ContextChunk(source_id=c.source_id, chunk_index=c.chunk_index, text=_strip_page_markers(c.text),
                             score=round(c.score, 4), metadata=c.metadata) for c in chunks]
    citations = _source_citations(chunks)
    warnings, knowledge_gap = _diagnose_sources(chunks)
    confidence = _confidence(chunks, payload.retrieval)
    if confidence < 0.35:
        knowledge_gap = True
        warnings.append("Evidence confidence is low; the assistant will avoid unsupported claims.")
    if knowledge_gap:
        answer = "I couldn't find sufficient evidence in the indexed documents to answer this question."
        return QueryResponse(answer=answer, grounded=False, confidence=confidence, knowledge_gap=True,
                             citations=citations, warnings=list(dict.fromkeys(warnings)), context=context,
                             count=len(context), generation={"provider": "none", "used_llm": False}, trace=trace)

    if not payload.generate:
        return QueryResponse(answer="Relevant evidence retrieved. Generation was disabled for this request.",
                             grounded=False, confidence=confidence, knowledge_gap=False, citations=citations,
                             warnings=list(dict.fromkeys(warnings)), context=context, count=len(context),
                             generation={"provider": "retrieval_only", "used_llm": False}, trace=trace)

    generation_context = [{"source_id": c.source_id, "chunk_index": c.chunk_index, "text": c.text, "metadata": c.metadata} for c in chunks]
    try:
        generation = answer_generator.generate(question, generation_context, [c.model_dump() for c in citations])
    except LLMError as exc:
        logger.warning("LLM generation failed; using extractive fallback: %s", exc)
        fallback = answer_generator.extractive(question, generation_context)
        generation = type("Fallback", (), {"answer": fallback, "provider": "extractive-fallback", "model": None,
                                            "latency_ms": 0.0, "used_llm": False})()
        warnings.append("LLM generation was unavailable; an extractive grounded answer was used.")
    grounded, citation_verification = _grounded_status(generation.answer, citations, context,
                                                        used_llm=generation.used_llm)
    return QueryResponse(answer=generation.answer, grounded=grounded, confidence=confidence, knowledge_gap=False,
                         citations=citations, warnings=list(dict.fromkeys(warnings)), context=context,
                         count=len(context), generation={"provider": generation.provider, "model": generation.model,
                                                         "latency_ms": generation.latency_ms, "used_llm": generation.used_llm,
                                                          "citation_verification": citation_verification},
                         trace=trace)


@app.get("/health")
def health() -> dict:
    data = kb.health()
    data.update({"service": app_config.service_name, "version": __version__,
                 "uptime_seconds": round(time.time() - _START_TIME, 2),
                 "llm_provider": app_config.llm_provider, "llm_enabled": answer_generator.enabled})
    return data


@app.get("/healthz")
def healthz() -> dict:
    return health()


@app.get("/", include_in_schema=False, response_model=None)
def root():
    return RedirectResponse(url="/ui/") if _UI_DIR.is_dir() else {"status": "ok", "service": app_config.service_name}


@app.get("/ready")
def ready() -> dict:
    if not kb.storage_probe():
        raise HTTPException(status_code=503, detail="Storage backend is not available.")
    return {"status": "ready", "service": app_config.service_name, "version": __version__, "storage": kb.storage_status()}


@app.get("/stats", dependencies=[Depends(_guard_api_key)])
def stats() -> dict:
    return kb.stats()


@app.get("/documents", response_model=DocumentsResponse, dependencies=[Depends(_guard_api_key)])
def documents(request: Request) -> DocumentsResponse:
    workspace_id = getattr(request.state, "workspace_id", None)
    items = [DocumentSummary(**item) for item in kb.list_documents(workspace_id=workspace_id)]
    return DocumentsResponse(documents=items, count=len(items))


@app.post("/ingest", response_model=IngestResponse, status_code=201, dependencies=[Depends(_guard_api_key)])
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    workspace_id = getattr(request.state, "workspace_id", None)
    try:
        result = kb.ingest_with_info(payload.source_id, payload.content, chunk_size=payload.chunk_size,
                                     chunk_overlap=payload.chunk_overlap, chunking_strategy=payload.chunking_strategy,
                                     metadata=payload.metadata, workspace_id=workspace_id)
        return IngestResponse(**result)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc


@app.post("/ingest/file", response_model=FileIngestResponse, status_code=201, dependencies=[Depends(_guard_api_key)])
async def ingest_file(request: Request, file: UploadFile = File(...),
                      source_id: str | None = Query(default=None, min_length=1, max_length=200),
                      chunk_size: int = Query(default=800, ge=1, le=4000),
                      chunk_overlap: int = Query(default=120, ge=0, le=4000)) -> FileIngestResponse:
    workspace_id = getattr(request.state, "workspace_id", None)
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size.")
    filename = file.filename or "document"
    if not Path(filename).suffix:
        raise HTTPException(status_code=400, detail="Uploaded file must have a supported extension.")
    raw = await _read_upload_limited(file, app_config.request_body_limit_bytes)
    if raw is None:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
    try:
        parsed = parse_document(filename, raw, source_id)
        stored_name = f"{parsed.metadata.get('content_hash', 'document')}{Path(filename).suffix.lower()}"
        parsed.metadata["stored_file"] = stored_name
        result = kb.ingest_with_info(parsed.source_id, parsed.content, chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap, chunking_strategy="smart",
                                     metadata=parsed.metadata, workspace_id=workspace_id)
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc
    # Persist the raw artifact only after indexing succeeded so a failed ingest
    # cannot leave orphaned files behind.
    try:
        upload_dir = Path(app_config.upload_dir).expanduser()
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored_name).write_bytes(raw)
    except OSError:
        logger.warning("Could not persist uploaded artifact for %s", parsed.source_id)
    return FileIngestResponse(**result, filename=filename, file_type=Path(filename).suffix.lstrip("."))


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes | None:
    """Read an upload in chunks, aborting as soon as the byte limit is exceeded.

    Returns ``None`` when the payload exceeds ``max_bytes`` so the caller can
    reject with 413 without ever buffering the full body in memory.
    """
    buffer = bytearray()
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        buffer.extend(piece)
        if len(buffer) > max_bytes:
            return None
    return bytes(buffer)


@app.post("/ingest/bulk", response_model=BulkIngestResponse, dependencies=[Depends(_guard_api_key)])
def ingest_bulk(payload: BulkIngestRequest) -> BulkIngestResponse:
    try:
        total = kb.ingest_many([document.model_dump() for document in payload.documents])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BulkIngestResponse(documents=len(payload.documents), chunks_indexed=total, total_chunks=kb.stats()["chunks"])


# --------------------------------------------------------------------------
# Background ingestion (Phase 1.8)
# --------------------------------------------------------------------------

_ingest_jobs: dict[str, dict[str, Any]] = {}


def _run_async_ingest(job_id: str, payload: IngestRequest) -> None:
    job = _ingest_jobs.setdefault(job_id, {"status": "processing", "source_id": payload.source_id,
                                            "error": None, "result": None})
    job["status"] = "processing"
    try:
        result = kb.ingest_with_info(payload.source_id, payload.content, chunk_size=payload.chunk_size,
                                     chunk_overlap=payload.chunk_overlap, chunking_strategy=payload.chunking_strategy,
                                     metadata=payload.metadata)
        job["status"] = "indexed"
        job["result"] = result
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)


@app.post("/ingest/async", response_model=IngestAsyncResponse, status_code=202, dependencies=[Depends(_guard_api_key)])
def ingest_async(payload: IngestRequest, background_tasks: BackgroundTasks) -> IngestAsyncResponse:
    """Queue a document for background ingestion and return a job id."""
    job_id = uuid.uuid4().hex[:16]
    _ingest_jobs[job_id] = {"status": "queued", "source_id": payload.source_id, "error": None, "result": None}
    background_tasks.add_task(_run_async_ingest, job_id, payload)
    return IngestAsyncResponse(job_id=job_id, status="queued")


@app.get("/ingest/status/{job_id}", response_model=IngestJobStatus, dependencies=[Depends(_guard_api_key)])
def ingest_status(job_id: str) -> IngestJobStatus:
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
    return IngestJobStatus(job_id=job_id, status=job["status"], source_id=job.get("source_id"),
                           error=job.get("error"), result=job.get("result"))


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_query(payload: QueryRequest, request: Request) -> Iterator[str]:
    """Yield SSE events for a streaming query: deltas, then a final done event."""
    workspace_id = getattr(request.state, "workspace_id", None)
    question = kb.augment_with_history(payload.session_id, payload.question)
    chunks, trace = kb.query_with_trace(question, top_k=payload.top_k, min_score=payload.min_score,
                                        retrieval=payload.retrieval, embedding_model=payload.embedding_model,
                                        embedding_provider=payload.embedding_provider,
                                        local_dimensions=payload.local_dimensions, reranker=payload.reranker,
                                        candidate_pool_size=payload.candidate_pool_size,
                                        metadata_filter=payload.metadata_filter, workspace_id=workspace_id)
    chunks = _trim_context(_dedup_context(chunks), payload.context_max_tokens)

    if not payload.generate or not answer_generator.enabled:
        response = _build_response(question, chunks, trace, payload)
        yield _sse_event({"type": "done", "response": response.model_dump()})
        return

    warnings, knowledge_gap = _diagnose_sources(chunks)
    confidence = _confidence(chunks, payload.retrieval)
    if confidence < 0.35:
        knowledge_gap = True
        warnings.append("Evidence confidence is low; the assistant will avoid unsupported claims.")

    if knowledge_gap:
        response = _build_response(question, chunks, trace, payload)
        yield _sse_event({"type": "done", "response": response.model_dump()})
        return

    context = [ContextChunk(source_id=c.source_id, chunk_index=c.chunk_index, text=_strip_page_markers(c.text),
                            score=round(c.score, 4), metadata=c.metadata) for c in chunks]
    citations = _source_citations(chunks)
    generation_context = [{"source_id": c.source_id, "chunk_index": c.chunk_index, "text": c.text,
                           "metadata": c.metadata} for c in chunks]
    deltas: list[str] = []
    used_llm = False
    try:
        for delta in answer_generator.generate_stream(question, generation_context, [c.model_dump() for c in citations]):
            if delta:
                deltas.append(delta)
                yield _sse_event({"type": "delta", "text": delta})
        used_llm = True
    except LLMError as exc:
        logger.warning("LLM stream failed; using extractive fallback: %s", exc)
    answer = "".join(deltas) if used_llm else answer_generator.extractive(question, generation_context)
    grounded, citation_verification = _grounded_status(answer, citations, context, used_llm=used_llm)
    response = QueryResponse(
        answer=answer, grounded=grounded, confidence=confidence, knowledge_gap=False,
        citations=citations, warnings=list(dict.fromkeys(warnings)), context=context, count=len(context),
        generation={"provider": answer_generator.provider, "model": answer_generator.model,
                    "used_llm": used_llm, "citation_verification": citation_verification},
        trace=trace,
    )
    yield _sse_event({"type": "done", "response": response.model_dump()})
    if payload.session_id:
        kb.record_turn(payload.session_id, "user", payload.question)
        kb.record_turn(payload.session_id, "assistant", response.answer)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_guard_api_key)])
def query(payload: QueryRequest, request: Request) -> QueryResponse | StreamingResponse:
    workspace_id = getattr(request.state, "workspace_id", None)
    if payload.stream:
        return StreamingResponse(_stream_query(payload, request), media_type="text/event-stream")
    try:
        question = kb.augment_with_history(payload.session_id, payload.question)
        chunks, trace = kb.query_with_trace(question, top_k=payload.top_k, min_score=payload.min_score,
                                            retrieval=payload.retrieval, embedding_model=payload.embedding_model,
                                            embedding_provider=payload.embedding_provider,
                                            local_dimensions=payload.local_dimensions, reranker=payload.reranker,
                                            candidate_pool_size=payload.candidate_pool_size,
                                            metadata_filter=payload.metadata_filter, workspace_id=workspace_id)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="Query processing failed. Check embedding configuration.") from exc
    if not chunks and kb.stats()["chunks"] == 0:
        raise HTTPException(status_code=404, detail="No documents are indexed. Upload or ingest a document first.")
    response = _build_response(question, chunks, trace, payload)
    if payload.session_id:
        kb.record_turn(payload.session_id, "user", payload.question)
        kb.record_turn(payload.session_id, "assistant", response.answer)
    return response


@app.get("/sessions", response_model=SessionsListResponse, dependencies=[Depends(_guard_api_key)])
def list_sessions() -> SessionsListResponse:
    sessions = kb.session_ids()
    return SessionsListResponse(sessions=sessions, count=len(sessions))


@app.delete("/sessions/{session_id}", response_model=SessionClearResponse, dependencies=[Depends(_guard_api_key)])
def clear_session(session_id: str) -> SessionClearResponse:
    removed = kb.clear_session(session_id)
    return SessionClearResponse(session_id=session_id, removed_turns=removed)


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(_guard_api_key)])
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Capture thumbs-up/down + comments for answers (3.8); feeds the eval datasets."""
    kb.add_feedback(payload.source_id, payload.question, payload.answer,
                    rating=payload.rating, comment=payload.comment)
    feedback = kb.list_feedback(source_id=payload.source_id, limit=1)
    return FeedbackResponse(feedback_id=feedback[0]["id"], created_at=feedback[0]["created_at"])


@app.get("/feedback", response_model=FeedbackListResponse, dependencies=[Depends(_guard_api_key)])
def list_feedback(source_id: str | None = None, limit: int = 20) -> FeedbackListResponse:
    rows = kb.list_feedback(source_id=source_id, limit=limit)
    return FeedbackListResponse(feedback=rows, count=len(rows))


@app.get("/workspaces", response_model=WorkspacesResponse, dependencies=[Depends(_guard_api_key)])
def list_workspaces() -> WorkspacesResponse:
    rows = kb.workspaces()
    return WorkspacesResponse(workspaces=[WorkspaceInfo(id=row[0], name=row[1]) for row in rows], count=len(rows))


@app.post("/auth/token", response_model=TokenResponse, dependencies=[Depends(_guard_api_key)])
def mint_token(payload: TokenRequest) -> TokenResponse:
    """Mint a JWT for a user + workspace (Phase 4.1). Requires a configured RAG_JWT_SECRET."""
    if not app_config.jwt_secret:
        raise HTTPException(status_code=400, detail="RAG_JWT_SECRET is not configured.")
    return TokenResponse(token=create_token(user_id=payload.user_id, workspace_id=payload.workspace_id,
                                            app_config=app_config))


@app.post("/query/semantic", response_model=QueryResponse, dependencies=[Depends(_guard_api_key)])
def query_semantic(payload: SemanticQueryRequest, request: Request) -> QueryResponse:
    return query(payload, request)


@app.delete("/documents/{source_id}", response_model=DeleteResponse, dependencies=[Depends(_guard_api_key)])
def delete_source(source_id: str, request: Request) -> DeleteResponse:
    workspace_id = getattr(request.state, "workspace_id", None)
    docs = {d["source_id"]: d for d in kb.list_documents(workspace_id=workspace_id)}
    document = docs.get(source_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    stored_file = (document.get("metadata") or {}).get("stored_file")
    removed = kb.remove_source(source_id, workspace_id=workspace_id)
    if stored_file:
        try:
            (Path(app_config.upload_dir).expanduser() / str(stored_file)).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stored source artifact for %s", source_id)
    return DeleteResponse(source_id=source_id, removed_chunks=removed, total_chunks=kb.stats()["chunks"])


@app.delete("/clear", dependencies=[Depends(_guard_api_key)])
def clear_all(x_clear_confirm: str | None = Header(default=None, alias="X-Clear-Confirm")) -> JSONResponse:
    if (x_clear_confirm or "").strip().upper() != "DELETE":
        raise HTTPException(status_code=400,
                            detail='Confirmation required: send header "X-Clear-Confirm: DELETE" to clear the index.')
    removed = kb.clear()
    return JSONResponse({"removed_chunks": removed, "remaining_chunks": kb.stats()["chunks"]})


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return _metrics.snapshot()


@app.post("/evals/run", response_model=EvalResponse, dependencies=[Depends(_guard_api_key)])
def evals(payload: EvalRequest) -> EvalResponse:
    if kb.stats()["chunks"] == 0:
        raise HTTPException(status_code=404, detail="No indexed chunks available. Ingest documents before running evaluations.")
    try:
        result = run_retrieval_eval(kb, [case.model_dump() for case in payload.cases])
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=503, detail="Evaluation failed.") from exc
    return EvalResponse(**result)
