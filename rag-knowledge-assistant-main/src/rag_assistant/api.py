from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Deque, Dict

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from rag_assistant import __version__
from rag_assistant.config import AppConfig
from rag_assistant.document_processor import DocumentProcessingError, parse_document
from rag_assistant.evals import run_retrieval_eval
from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.llm import AnswerGenerator, LLMError
from rag_assistant.models import (
    BulkIngestRequest,
    BulkIngestResponse,
    Citation,
    ContextChunk,
    DeleteResponse,
    DocumentsResponse,
    DocumentSummary,
    EvalRequest,
    EvalResponse,
    ErrorResponse,
    FileIngestResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RouteMetrics,
    SemanticQueryRequest,
)


app_config = AppConfig.from_env()
kb = KnowledgeBase(
    chunk_size=app_config.default_chunk_size,
    chunk_overlap=app_config.default_chunk_overlap,
    storage_path=app_config.storage_path,
    default_embedding_model=app_config.default_embedding_model,
)
answer_generator = AnswerGenerator(
    provider=app_config.llm_provider,
    model=app_config.llm_model,
    api_key=app_config.llm_api_key,
    base_url=app_config.llm_base_url,
    timeout_seconds=app_config.llm_timeout_seconds,
)

logging.basicConfig(level=getattr(logging, app_config.log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag-knowledge-assistant")

app = FastAPI(
    title="TrustAware RAG Knowledge Assistant",
    version=__version__,
    description=("Document-grounded RAG API with hybrid retrieval, reranking, citations, "
                 "confidence-aware no-answer handling, and knowledge diagnostics."),
    docs_url="/docs" if app_config.docs_enabled else None,
    redoc_url="/redoc" if app_config.docs_enabled else None,
    openapi_url="/openapi.json" if app_config.docs_enabled else None,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UI_DIR = _PROJECT_ROOT / "ui"
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
        self._metrics: Dict[str, _RouteMetric] = {}
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
        self._timestamps: Dict[str, Deque[float]] = defaultdict(deque)
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


def _client_key(request: Request) -> str:
    if app_config.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client and request.client.host else "anonymous"


def _rate_limit_exempt(path: str) -> bool:
    return path in {"/health", "/healthz", "/ready", "/metrics"} or path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json"


@app.middleware("http")
async def middleware(request: Request, call_next):
    started = time.perf_counter()
    request.state.request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    path = request.url.path

    if not _rate_limit_exempt(path) and app_config.rate_limit_requests > 0:
        allowed, remaining, reset_after = _limiter.allow(_client_key(request))
        if not allowed:
            payload = _error_payload(request, 429, "Rate limit exceeded. Retry later.")
            response = JSONResponse(status_code=429, content=payload.model_dump())
            response.headers.update({
                "x-ratelimit-limit": str(app_config.rate_limit_requests),
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset-after": str(reset_after),
                "x-request-id": request.state.request_id,
                "Retry-After": str(reset_after),
            })
            _metrics.record(request.method, path, 429, 0.0)
            return response
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset_after = reset_after

    content_length = request.headers.get("content-length")
    if content_length and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        try:
            if int(content_length) > app_config.request_body_limit_bytes:
                raise HTTPException(status_code=413, detail="Request body exceeds the configured limit.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc

    try:
        response = await call_next(request)
    except HTTPException as exc:
        _metrics.record(request.method, path, exc.status_code, (time.perf_counter() - started) * 1000)
        raise
    except Exception:
        _metrics.record(request.method, path, 500, (time.perf_counter() - started) * 1000)
        logger.exception("Unhandled request failure: %s %s", request.method, path)
        raise

    _metrics.record(request.method, path, response.status_code, (time.perf_counter() - started) * 1000)
    response.headers["x-request-id"] = request.state.request_id
    if not _rate_limit_exempt(path):
        response.headers["x-ratelimit-limit"] = str(app_config.rate_limit_requests)
        response.headers["x-ratelimit-remaining"] = str(getattr(request.state, "rate_limit_remaining", 0))
        response.headers["x-ratelimit-reset-after"] = str(getattr(request.state, "rate_limit_reset_after", 0))
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["x-frame-options"] = "DENY"
    return response


async def _guard_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    if app_config.api_key and x_api_key != app_config.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.", headers={"WWW-Authenticate": "ApiKey"})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    payload = _error_payload(request, exc.status_code, str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


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
        page_match = re.findall(r"\[PAGE\s+(\d+)\]", chunk.text, flags=re.I)
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
    for i, (source_a, nums_a, terms_a) in enumerate(signatures):
        for source_b, nums_b, terms_b in signatures[i + 1:]:
            if nums_a and nums_b and nums_a != nums_b and len(terms_a & terms_b) >= 4:
                warnings.append("Potential conflicting information detected across retrieved sources. Review the cited documents.")
                return warnings, False
    return warnings, False


def _build_response(question: str, chunks: list, trace: dict, payload: QueryRequest) -> QueryResponse:
    context = [ContextChunk(source_id=c.source_id, chunk_index=c.chunk_index, text=c.text,
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
    return QueryResponse(answer=generation.answer, grounded=True, confidence=confidence, knowledge_gap=False,
                         citations=citations, warnings=list(dict.fromkeys(warnings)), context=context,
                         count=len(context), generation={"provider": generation.provider, "model": generation.model,
                                                         "latency_ms": generation.latency_ms, "used_llm": generation.used_llm},
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
def documents() -> DocumentsResponse:
    items = [DocumentSummary(**item) for item in kb.list_documents()]
    return DocumentsResponse(documents=items, count=len(items))


@app.post("/ingest", response_model=IngestResponse, status_code=201, dependencies=[Depends(_guard_api_key)])
def ingest(payload: IngestRequest) -> IngestResponse:
    try:
        result = kb.ingest_with_info(payload.source_id, payload.content, chunk_size=payload.chunk_size,
                                     chunk_overlap=payload.chunk_overlap, chunking_strategy=payload.chunking_strategy,
                                     metadata=payload.metadata)
        return IngestResponse(**result)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc


@app.post("/ingest/file", response_model=FileIngestResponse, status_code=201, dependencies=[Depends(_guard_api_key)])
async def ingest_file(file: UploadFile = File(...), source_id: str | None = None,
                      chunk_size: int = 800, chunk_overlap: int = 120) -> FileIngestResponse:
    filename = file.filename or "document"
    if not Path(filename).suffix:
        raise HTTPException(status_code=400, detail="Uploaded file must have a supported extension.")
    raw = await file.read()
    if len(raw) > app_config.request_body_limit_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
    try:
        parsed = parse_document(filename, raw, source_id)
        upload_dir = Path(app_config.upload_dir).expanduser()
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{parsed.metadata.get('content_hash', 'document')}{Path(filename).suffix.lower()}"
        (upload_dir / stored_name).write_bytes(raw)
        parsed.metadata["stored_file"] = stored_name
        result = kb.ingest_with_info(parsed.source_id, parsed.content, chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap, chunking_strategy="smart",
                                     metadata=parsed.metadata)
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=503, detail="Document indexing failed.") from exc
    return FileIngestResponse(**result, filename=filename, file_type=Path(filename).suffix.lstrip("."))


@app.post("/ingest/bulk", response_model=BulkIngestResponse, dependencies=[Depends(_guard_api_key)])
def ingest_bulk(payload: BulkIngestRequest) -> BulkIngestResponse:
    total = 0
    for document in payload.documents:
        try:
            total += kb.ingest(document.source_id, document.content, chunk_size=document.chunk_size,
                               chunk_overlap=document.chunk_overlap, chunking_strategy=document.chunking_strategy,
                               metadata=document.metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BulkIngestResponse(documents=len(payload.documents), chunks_indexed=total, total_chunks=kb.stats()["chunks"])


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_guard_api_key)])
def query(payload: QueryRequest) -> QueryResponse:
    try:
        chunks, trace = kb.query_with_trace(payload.question, top_k=payload.top_k, min_score=payload.min_score,
                                            retrieval=payload.retrieval, embedding_model=payload.embedding_model,
                                            embedding_provider=payload.embedding_provider,
                                            local_dimensions=payload.local_dimensions, reranker=payload.reranker,
                                            candidate_pool_size=payload.candidate_pool_size,
                                            metadata_filter=payload.metadata_filter)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="Query processing failed. Check embedding configuration.") from exc
    if not chunks and kb.stats()["chunks"] == 0:
        raise HTTPException(status_code=404, detail="No documents are indexed. Upload or ingest a document first.")
    return _build_response(payload.question, chunks, trace, payload)


@app.post("/query/semantic", response_model=QueryResponse, dependencies=[Depends(_guard_api_key)])
def query_semantic(payload: SemanticQueryRequest) -> QueryResponse:
    return query(payload)


@app.delete("/documents/{source_id}", response_model=DeleteResponse, dependencies=[Depends(_guard_api_key)])
def delete_source(source_id: str) -> DeleteResponse:
    docs = {d["source_id"]: d for d in kb.list_documents()}
    stored_file = (docs.get(source_id) or {}).get("metadata", {}).get("stored_file")
    removed = kb.remove_source(source_id)
    if stored_file:
        try:
            (Path(app_config.upload_dir).expanduser() / str(stored_file)).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stored source artifact for %s", source_id)
    return DeleteResponse(source_id=source_id, removed_chunks=removed, total_chunks=kb.stats()["chunks"])


@app.delete("/clear", dependencies=[Depends(_guard_api_key)])
def clear_all() -> JSONResponse:
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
