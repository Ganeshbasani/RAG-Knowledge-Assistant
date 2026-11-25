from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ChunkingStrategy = Literal["tokens", "sentence", "paragraph", "smart"]
RetrievalMode = Literal["tfidf", "semantic", "hybrid"]
RerankerMode = Literal["none", "term_overlap", "cross_encoder"]

# Serialized metadata is persisted in the index file; keep individual entries bounded.
_MAX_METADATA_BYTES = 64 * 1024


def _validate_metadata_size(value: dict[str, Any]) -> dict[str, Any]:
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-serializable") from exc
    if len(serialized.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the 64KB serialized size limit")
    return value


class IngestRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=2_000_000)
    chunk_size: int = Field(default=800, ge=1, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=4000)
    chunking_strategy: ChunkingStrategy = "smart"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def _strip_source_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_id cannot be empty")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _valid_overlap(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 800)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return value

    @field_validator("metadata")
    @classmethod
    def _valid_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_metadata_size(value)


class IngestResponse(BaseModel):
    source_id: str
    chunks_indexed: int
    total_chunks: int
    replaced_existing: bool = False
    content_hash: str | None = None


class FileIngestResponse(IngestResponse):
    filename: str
    file_type: str


class BulkIngestRequest(BaseModel):
    documents: list[IngestRequest] = Field(min_length=1, max_length=100)


class BulkIngestResponse(BaseModel):
    documents: int
    chunks_indexed: int
    total_chunks: int


class IngestAsyncResponse(BaseModel):
    job_id: str
    status: str


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    source_id: str | None = None
    error: str | None = None
    result: IngestResponse | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.08, ge=0.0, le=1.0)
    retrieval: RetrievalMode = "hybrid"
    embedding_model: str | None = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers", "local", "local_hash", "local_tfidf", "onnx_local"
    ] = "local"
    local_dimensions: int = Field(default=256, ge=8, le=4096)
    reranker: RerankerMode = "term_overlap"
    candidate_pool_size: int = Field(default=12, ge=1, le=100)
    generate: bool = True
    context_max_tokens: int = Field(default=0, ge=0, le=8000,
                                    description="Token budget for assembled context (0 = no limit).")
    stream: bool = False
    session_id: str | None = Field(
        default=None,
        max_length=120,
        description="Session id for conversational memory; accepted but ignored when no history exists.",
    )
    metadata_filter: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value

    @field_validator("metadata_filter")
    @classmethod
    def _valid_metadata_filter(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_metadata_size(value)


class SemanticQueryRequest(QueryRequest):
    retrieval: Literal["semantic"] = "semantic"


class Citation(BaseModel):
    citation_id: int
    source_id: str
    chunk_index: int
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextChunk(BaseModel):
    source_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float | None = Field(
        default=None,
        description="Reserved for calibrated confidence; null until confidence is evaluated against labeled data.",
    )
    knowledge_gap: bool
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    context: list[ContextChunk]
    count: int
    generation: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    source_id: str
    chunks: int
    filename: str | None = None
    file_type: str | None = None
    content_hash: str | None = None
    ingested_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    count: int


class DeleteResponse(BaseModel):
    source_id: str
    removed_chunks: int
    total_chunks: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None
    path: str | None = None
    details: dict[str, Any] | None = None


class RouteMetrics(BaseModel):
    requests: int
    errors: int
    avg_response_ms: float
    last_status: int | None = None


class MetricsResponse(BaseModel):
    timestamp: str
    uptime_seconds: float
    requests_total: int
    errors_total: int
    routes: dict[str, RouteMetrics]


class EvalCase(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    expected_source_ids: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.08, ge=0.0, le=1.0)
    retrieval: RetrievalMode = "hybrid"
    embedding_model: str | None = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers", "local", "local_hash", "local_tfidf", "onnx_local"
    ] = "local"
    local_dimensions: int = Field(default=256, ge=8, le=4096)
    reranker: RerankerMode = "term_overlap"
    candidate_pool_size: int = Field(default=12, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def _strip_eval_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


class FeedbackRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=5000)
    answer: str = Field(min_length=1, max_length=10_000)
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    feedback_id: int
    created_at: str


class FeedbackListResponse(BaseModel):
    feedback: list[dict[str, Any]]
    count: int


class WorkspaceInfo(BaseModel):
    id: str
    name: str


class WorkspacesResponse(BaseModel):
    workspaces: list[WorkspaceInfo]
    count: int


class TokenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    workspace_id: str = Field(min_length=1, max_length=120)


class TokenResponse(BaseModel):
    token: str


class BrowserSessionRequest(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class AuthConfigResponse(BaseModel):
    auth_required: bool
    browser_login_enabled: bool


class SessionsListResponse(BaseModel):
    sessions: list[str]
    count: int


class SessionClearResponse(BaseModel):
    session_id: str
    removed_turns: int


class EvalRequest(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=200)


class EvalCaseResult(BaseModel):
    question: str
    expected_source_ids: list[str]
    returned_source_ids: list[str]
    matched: bool
    reciprocal_rank: float
    top_hit_source_id: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


class EvalResponse(BaseModel):
    cases: int
    hits: int
    hit_rate: float
    mrr: float
    results: list[EvalCaseResult]
