from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


ChunkingStrategy = Literal["tokens", "sentence", "paragraph", "smart"]
RetrievalMode = Literal["tfidf", "semantic", "hybrid"]
RerankerMode = Literal["none", "term_overlap"]


class IngestRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=2_000_000)
    chunk_size: int = Field(default=800, ge=1, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=4000)
    chunking_strategy: ChunkingStrategy = "smart"
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    documents: List[IngestRequest] = Field(min_length=1, max_length=100)


class BulkIngestResponse(BaseModel):
    documents: int
    chunks_indexed: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.08, ge=0.0, le=1.0)
    retrieval: RetrievalMode = "hybrid"
    embedding_model: Optional[str] = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers", "local", "local_hash", "local_tfidf", "onnx_local"
    ] = "local_tfidf"
    local_dimensions: int = Field(default=256, ge=8, le=4096)
    reranker: RerankerMode = "term_overlap"
    candidate_pool_size: int = Field(default=12, ge=1, le=100)
    generate: bool = True
    session_id: str | None = Field(default=None, max_length=120)
    metadata_filter: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


class SemanticQueryRequest(QueryRequest):
    retrieval: Literal["semantic"] = "semantic"


class Citation(BaseModel):
    citation_id: int
    source_id: str
    chunk_index: int
    label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextChunk(BaseModel):
    source_id: str
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    knowledge_gap: bool
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    context: List[ContextChunk]
    count: int
    generation: Dict[str, Any] = Field(default_factory=dict)
    trace: Dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    source_id: str
    chunks: int
    filename: str | None = None
    file_type: str | None = None
    content_hash: str | None = None
    ingested_at: str | None = None
    updated_at: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentsResponse(BaseModel):
    documents: List[DocumentSummary]
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
    details: Dict[str, Any] | None = None


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
    routes: Dict[str, RouteMetrics]


class EvalCase(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    expected_source_ids: List[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.08, ge=0.0, le=1.0)
    retrieval: RetrievalMode = "hybrid"
    embedding_model: Optional[str] = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers", "local", "local_hash", "local_tfidf", "onnx_local"
    ] = "local_tfidf"
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


class EvalRequest(BaseModel):
    cases: List[EvalCase] = Field(min_length=1, max_length=200)


class EvalCaseResult(BaseModel):
    question: str
    expected_source_ids: List[str]
    returned_source_ids: List[str]
    matched: bool
    reciprocal_rank: float
    top_hit_source_id: str | None = None
    trace: Dict[str, Any] = Field(default_factory=dict)


class EvalResponse(BaseModel):
    cases: int
    hits: int
    hit_rate: float
    mrr: float
    results: List[EvalCaseResult]
