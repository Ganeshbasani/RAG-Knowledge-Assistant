from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from rag_assistant.embeddings import EmbeddingProviderError, ProviderSpec, build_embedding_provider
from rag_assistant.metadata import DocumentRecord, MetadataStore, create_metadata_store
from rag_assistant.ranking import BM25Index
from rag_assistant.rerankers import CrossEncoderReranker
from rag_assistant.sessions import SessionStore
from rag_assistant.stores import (
    STORAGE_SCHEMA_VERSION,
    JSONVectorStore,
    VectorStore,
)

# Re-exported for backwards compatibility (error types live in stores).
from rag_assistant.stores import (
    KnowledgeBaseError as KnowledgeBaseError,
)
from rag_assistant.stores import (
    KnowledgeBaseStorageError as KnowledgeBaseStorageError,
)

TOKEN_PATTERN = r"[a-zA-Z0-9]+"
_TOKEN_RE = re.compile(TOKEN_PATTERN)
SENTENCE_PATTERN = r"(?<=[.!?])\s+"
PARAGRAPH_PATTERN = r"\n\s*\n+"
_SENTENCE_RE = re.compile(SENTENCE_PATTERN)
_PARAGRAPH_RE = re.compile(PARAGRAPH_PATTERN)
_BM25_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "should", "so", "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
})
SUPPORTED_CHUNKING_STRATEGIES = {"tokens", "sentence", "paragraph", "smart"}
SUPPORTED_RETRIEVAL_MODES = {"tfidf", "semantic", "hybrid"}
SUPPORTED_RERANKERS = {"none", "term_overlap", "cross_encoder"}


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]


class KnowledgeBase:
    """Thread-safe document index with lexical, semantic and hybrid retrieval.

    Storage is split into two seams:

    - ``_store`` (:class:`VectorStore`) owns chunk records and embeddings;
      the default is the zero-dependency JSON backend, Qdrant is opt-in.
    - ``_metadata_store`` (:class:`MetadataStore`) owns document metadata
      (SQLite by default, PostgreSQL-ready).

    The public methods (``ingest_with_info``, ``query_with_trace``, ...) are
    the compatibility seam: they are stable across storage backends.
    """

    STORAGE_SCHEMA_VERSION = STORAGE_SCHEMA_VERSION

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        storage_path: str | None = None,
        default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store: VectorStore | None = None,
        metadata_store: MetadataStore | None = None,
        reranker: str = "term_overlap",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        session_store: SessionStore | None = None,
        query_rewriter: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._validate_chunk_config(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._storage_path = Path(storage_path).expanduser() if storage_path else None
        self._lock = RLock()
        self._store = (
            vector_store if vector_store is not None
            else JSONVectorStore(str(self._storage_path) if self._storage_path else None)
        )
        self._metadata_store = (
            metadata_store if metadata_store is not None
            else create_metadata_store(storage_path=str(self._storage_path) if self._storage_path else None)
        )
        self._default_embedding_model = (default_embedding_model or "sentence-transformers/all-MiniLM-L6-v2").strip()
        self._embedding_provider = None
        self._provider_cache_key = None
        self._reranker_name = (reranker or "term_overlap").strip().lower()
        self._cross_encoder = CrossEncoderReranker(model=reranker_model) if self._reranker_name == "cross_encoder" else None
        self._session_store = session_store if session_store is not None else SessionStore()
        self._query_rewriter = query_rewriter

        # In-memory lexical/semantic state, loaded from the store.
        self._chunks: list[dict[str, Any]] = self._store.load_chunks()
        self._vectors: dict[tuple[str, int], list[float]] = self._store.load_vectors()
        self._vector_key: tuple[str, str | None, int | None] | None = self._store.load_vector_key()
        self._bm25 = BM25Index()
        self._bm25.rebuild({(c["source_id"], c["chunk_index"]): KnowledgeBase._tokenize(c["text"]) for c in self._chunks})
        self._sync_document_metadata()

    def _sync_document_metadata(self) -> None:
        """Reconcile the document catalog against the indexed chunk store at startup."""
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in self._chunks:
            by_source[chunk["source_id"]].append(chunk)
        stored = {record.source_id for record in self._metadata_store.list_documents()}
        for source_id in stored - by_source.keys():
            self._metadata_store.delete_document(source_id)
        for source_id, chunks in by_source.items():
            metadata = chunks[0].get("metadata", {}) or {}
            self._metadata_store.upsert_document(
                DocumentRecord.from_chunk_metadata(source_id, len(chunks), metadata)
            )

    # ------------------------------------------------------------------ validators
    @staticmethod
    def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    @staticmethod
    def _validate_chunking_strategy(strategy: str) -> str:
        normalized = strategy.strip().lower()
        if normalized not in SUPPORTED_CHUNKING_STRATEGIES:
            raise ValueError("chunking_strategy must be one of: tokens, sentence, paragraph, smart")
        return normalized

    @staticmethod
    def _validate_query_config(retrieval: str, reranker: str) -> tuple[str, str]:
        normalized_retrieval = retrieval.strip().lower()
        normalized_reranker = reranker.strip().lower()
        if normalized_retrieval not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError("retrieval must be one of: tfidf, semantic, hybrid")
        if normalized_reranker not in SUPPORTED_RERANKERS:
            raise ValueError("reranker must be one of: none, term_overlap, cross_encoder")
        return normalized_retrieval, normalized_reranker

    # ----------------------------------------------------------------- text utils
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(KnowledgeBase._normalize(text).lower())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = KnowledgeBase._normalize(text)
        if not normalized:
            return []
        return [segment.strip() for segment in _SENTENCE_RE.split(normalized) if segment.strip()] or [normalized]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        paragraphs = [KnowledgeBase._normalize(block) for block in _PARAGRAPH_RE.split(text) if block.strip()]
        return [paragraph for paragraph in paragraphs if paragraph]

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON-like object")
        import json

        try:
            json.dumps(metadata)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        return dict(metadata)

    # ------------------------------------------------------------------ chunking
    def _split_chunks_tokens(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        tokens = _TOKEN_RE.findall(self._normalize(text))
        if not tokens:
            return []
        stride = chunk_size - chunk_overlap
        chunks: list[str] = []
        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start:start + chunk_size]
            if chunk_tokens:
                chunks.append(" ".join(chunk_tokens))
            if start + chunk_size >= len(tokens):
                break
        return chunks

    def _chunk_units(self, units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        chunks: list[str] = []
        current_units: list[str] = []
        current_token_count = 0
        for unit in units:
            clean_unit = self._normalize(unit)
            unit_tokens = self._tokenize(clean_unit)
            if not unit_tokens:
                continue
            if len(unit_tokens) >= chunk_size:
                if current_units:
                    chunks.append(self._normalize(" ".join(current_units)))
                    current_units, current_token_count = [], 0
                chunks.extend(self._split_chunks_tokens(clean_unit, chunk_size, chunk_overlap))
                continue
            if current_units and current_token_count + len(unit_tokens) > chunk_size:
                chunks.append(self._normalize(" ".join(current_units)))
                overlap_units: list[str] = []
                overlap_tokens = 0
                if chunk_overlap > 0:
                    for previous_unit in reversed(current_units):
                        previous_tokens = self._tokenize(previous_unit)
                        overlap_units.insert(0, previous_unit)
                        overlap_tokens += len(previous_tokens)
                        if overlap_tokens >= chunk_overlap:
                            break
                current_units = overlap_units
                current_token_count = sum(len(self._tokenize(item)) for item in current_units)
            current_units.append(clean_unit)
            current_token_count += len(unit_tokens)
        if current_units:
            chunks.append(self._normalize(" ".join(current_units)))
        return chunks

    def _split_chunks(self, text: str, chunk_size: int, chunk_overlap: int, chunking_strategy: str) -> list[str]:
        strategy = self._validate_chunking_strategy(chunking_strategy)
        if strategy == "tokens":
            return self._split_chunks_tokens(text, chunk_size, chunk_overlap)
        if strategy == "sentence":
            return self._chunk_units(self._split_sentences(text), chunk_size, chunk_overlap)
        paragraphs = self._split_paragraphs(text)
        if strategy == "paragraph":
            return self._chunk_units(paragraphs, chunk_size, chunk_overlap)
        smart_units: list[str] = []
        for paragraph in paragraphs or [text]:
            if len(self._tokenize(paragraph)) <= max(1, chunk_size // 2):
                smart_units.append(paragraph)
            else:
                smart_units.extend(self._split_sentences(paragraph))
        return self._chunk_units(smart_units, chunk_size, chunk_overlap)

    # -------------------------------------------------------------- ingest prepare
    @staticmethod
    def _workspace_filter(workspace_id: str | None, existing: dict[str, Any] | None) -> dict[str, Any] | None:
        """Server-enforced workspace filter (Phase 4.1): every workspace-scoped
        operation adds an exact-match ``workspace_id`` metadata requirement."""
        merged = dict(existing or {})
        if workspace_id:
            merged["workspace_id"] = workspace_id
        return merged or None

    def _prepare_ingest(self, source_id: str, content: str, *, chunk_size: int | None = None,
                        chunk_overlap: int | None = None, chunking_strategy: str = "smart",
                        metadata: dict[str, Any] | None = None,
                        workspace_id: str | None = None) -> tuple[str, list[str], dict[str, Any], str]:
        """Validate and chunk a document without touching any index state (Phase 0)."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        normalized_source_id = source_id.strip()
        safe_metadata = self._normalize_metadata(metadata)
        if workspace_id:
            safe_metadata["workspace_id"] = workspace_id
        chunk_size = self._chunk_size if chunk_size is None else chunk_size
        chunk_overlap = self._chunk_overlap if chunk_overlap is None else chunk_overlap
        self._validate_chunk_config(chunk_size, chunk_overlap)
        normalized_chunking_strategy = self._validate_chunking_strategy(chunking_strategy)
        chunks = self._split_chunks(content, chunk_size, chunk_overlap, normalized_chunking_strategy)
        if not chunks:
            raise ValueError("content did not produce any searchable chunks")
        return normalized_source_id, chunks, safe_metadata, normalized_chunking_strategy

    def _build_indexed_chunks(self, source_id: str, chunks: list[str], safe_metadata: dict[str, Any],
                              chunking_strategy: str) -> list[dict[str, Any]]:
        indexed: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = dict(safe_metadata)
            chunk_metadata.setdefault("chunking_strategy", chunking_strategy)
            chunk_metadata.setdefault("token_count", len(self._tokenize(chunk_text)))
            indexed.append({
                "source_id": source_id,
                "chunk_index": idx,
                "text": chunk_text,
                "metadata": chunk_metadata,
            })
        return indexed

    def _add_chunks_to_index(self, chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks:
            key = (chunk["source_id"], chunk["chunk_index"])
            self._bm25.add(key, KnowledgeBase._tokenize(chunk["text"]))

    def _remove_chunks_from_index(self, source_id: str) -> None:
        self._bm25.remove_source(source_id)

    # ------------------------------------------------------------------ ingest
    def ingest_with_info(self, source_id: str, content: str, *, chunk_size: int | None = None,
                         chunk_overlap: int | None = None, chunking_strategy: str = "smart",
                         metadata: dict[str, Any] | None = None, replace_existing: bool = True,
                         workspace_id: str | None = None) -> dict[str, Any]:
        normalized_source_id, chunks, safe_metadata, normalized_chunking_strategy = self._prepare_ingest(
            source_id, content, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy, metadata=metadata, workspace_id=workspace_id)

        with self._lock:
            existing = any(chunk["source_id"] == normalized_source_id for chunk in self._chunks)
            if replace_existing:
                self._chunks = [chunk for chunk in self._chunks if chunk["source_id"] != normalized_source_id]
                self._remove_chunks_from_index(normalized_source_id)
            indexed = self._build_indexed_chunks(normalized_source_id, chunks, safe_metadata, normalized_chunking_strategy)
            self._chunks.extend(indexed)
            self._add_chunks_to_index(indexed)
            self._store.save_chunks(self._chunks)
            self._metadata_store.upsert_document(
                DocumentRecord.from_chunk_metadata(normalized_source_id, len(indexed), indexed[0]["metadata"]))
            return {
                "source_id": normalized_source_id,
                "chunks_indexed": len(indexed),
                "total_chunks": len(self._chunks),
                "replaced_existing": existing and replace_existing,
                "content_hash": safe_metadata.get("content_hash"),
            }

    def ingest_many(self, documents: list[dict[str, Any]]) -> int:
        """Atomically ingest several documents; returns total chunks indexed.

        Every document is validated and chunked before any index state is
        mutated. If any document is invalid a ``ValueError`` is raised and the
        index (and its persistence file) is left exactly as it was.
        """
        prepared = [
            self._prepare_ingest(
                str(document.get("source_id") or ""),
                str(document.get("content") or ""),
                chunk_size=document.get("chunk_size"),
                chunk_overlap=document.get("chunk_overlap"),
                chunking_strategy=str(document.get("chunking_strategy") or "smart"),
                metadata=document.get("metadata"),
            )
            for document in documents
        ]
        if not prepared:
            return 0
        committed: list[tuple[str, list[dict[str, Any]]]] = []
        with self._lock:
            total = 0
            for normalized_source_id, chunks, safe_metadata, normalized_chunking_strategy in prepared:
                self._chunks = [chunk for chunk in self._chunks if chunk["source_id"] != normalized_source_id]
                self._remove_chunks_from_index(normalized_source_id)
                indexed = self._build_indexed_chunks(normalized_source_id, chunks, safe_metadata, normalized_chunking_strategy)
                self._chunks.extend(indexed)
                self._add_chunks_to_index(indexed)
                committed.append((normalized_source_id, indexed))
                total += len(indexed)
            self._store.save_chunks(self._chunks)
            for normalized_source_id, indexed in committed:
                self._metadata_store.upsert_document(
                    DocumentRecord.from_chunk_metadata(normalized_source_id, len(indexed), indexed[0]["metadata"]))
            return total

    def ingest(self, source_id: str, content: str, *, chunk_size: int | None = None,
               chunk_overlap: int | None = None, chunking_strategy: str = "smart",
               metadata: dict[str, Any] | None = None) -> int:
        return int(self.ingest_with_info(source_id, content, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                         chunking_strategy=chunking_strategy, metadata=metadata)["chunks_indexed"])

    # --------------------------------------------------------------- maintenance
    @staticmethod
    def _matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        return all(
            metadata.get(key) in value if isinstance(value, (list, tuple, set)) else metadata.get(key) == value
            for key, value in metadata_filter.items()
        )

    def remove_source(self, source_id: str, *, workspace_id: str | None = None) -> int:
        with self._lock:
            normalized = source_id.strip()
            if workspace_id:
                # Strict isolation: only delete the source if the caller's workspace matches.
                if not any(chunk["source_id"] == normalized and chunk["metadata"].get("workspace_id") == workspace_id
                           for chunk in self._chunks):
                    return 0
            before = len(self._chunks)
            self._chunks = [chunk for chunk in self._chunks
                            if not (chunk["source_id"] == normalized
                                    and (not workspace_id or chunk["metadata"].get("workspace_id") == workspace_id))]
            removed = before - len(self._chunks)
            self._store.delete_source(normalized)
            self._remove_chunks_from_index(normalized)
            self._metadata_store.delete_document(normalized)
            self._metadata_store.add_audit_event("remove_source", target_id=normalized, status="ok")
            return removed

    def clear(self) -> int:
        with self._lock:
            removed = len(self._chunks)
            self._chunks = []
            self._vectors = {}
            self._store.delete_all()
            self._bm25.rebuild({})
            self._metadata_store.delete_all_documents()
            self._metadata_store.add_audit_event("clear_all", status="ok")
            return removed

    # ------------------------------------------------------------ sessions (2.3)
    def augment_with_history(self, session_id: str | None, question: str) -> str:
        """Inject the last turns of the session into the question (empty history = passthrough)."""
        if not session_id:
            return question
        history = self._session_store.history(session_id)
        if not history:
            return question
        lines = ["[Conversation history — earlier turns in this session]"]
        lines += [f"{turn['role']}: {turn['content']}" for turn in history]
        lines.append(f"New question: {question}")
        return "\n".join(lines)

    def record_turn(self, session_id: str | None, role: str, content: str) -> None:
        self._session_store.append(session_id or "", role, content)

    def session_ids(self) -> list[str]:
        return self._session_store.ids()

    def clear_session(self, session_id: str) -> int:
        return self._session_store.clear(session_id)

    def add_feedback(self, source_id: str, question: str, answer: str, rating: int | None = None,
                     comment: str | None = None) -> None:
        """Persist user feedback for a retrieved answer (Phase 3.8)."""
        self._metadata_store.add_feedback(source_id, question, answer, rating=rating, comment=comment)

    def list_feedback(self, source_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self._metadata_store.list_feedback(source_id=source_id, limit=limit)

    def list_documents(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        records = self._metadata_store.list_documents()
        if workspace_id:
            records = [record for record in records if record.workspace_id == workspace_id]
        return [record.as_summary() for record in records]

    @staticmethod
    def is_document_overview_question(question: str) -> bool:
        action = re.search(r"\b(summar(?:y|ize|ise)|overview|explain|exaplian|describe)\b", question, re.IGNORECASE)
        target = re.search(r"\b(doc(?:ument)?s?|files?|library|attachments?)\b", question, re.IGNORECASE)
        return bool(action and target)

    def document_overview_chunks(self, *, top_k: int = 5, metadata_filter: dict[str, Any] | None = None,
                                 workspace_id: str | None = None) -> list[RetrievedChunk]:
        """Return one leading passage per matching document for broad summaries.

        Generic overview questions have few useful search terms. Sampling each
        matching document avoids making the summary depend on an arbitrary
        semantic-score threshold or one accidental nearest neighbor.
        """
        filters = self._workspace_filter(workspace_id, metadata_filter)
        with self._lock:
            selected: list[RetrievedChunk] = []
            seen_sources: set[str] = set()
            for chunk in self._chunks:
                source_id = chunk["source_id"]
                if source_id in seen_sources or not self._matches_filter(chunk.get("metadata", {}), filters):
                    continue
                seen_sources.add(source_id)
                selected.append(RetrievedChunk(source_id, chunk["chunk_index"], chunk["text"], 1.0,
                                               chunk.get("metadata", {})))
                if len(selected) >= max(1, min(top_k, 20)):
                    break
            return selected

    def workspaces(self) -> list[tuple[str, str]]:
        return self._metadata_store.list_workspaces()

    # --------------------------------------------------------------- persistence
    @property
    def storage_path(self) -> str | None:
        return str(self._storage_path) if self._storage_path else None

    @property
    def persistent(self) -> bool:
        return self._storage_path is not None

    def storage_status(self) -> dict[str, Any]:
        return {
            "enabled": self._storage_path is not None,
            "path": str(self._storage_path) if self._storage_path else None,
            "store_kind": getattr(self._store, "kind", type(self._store).__name__),
        }

    def storage_probe(self) -> bool:
        return self._store.storage_probe()

    # ------------------------------------------------------------ embeddings
    def _build_provider_cache_key(self, embedding_provider: str, embedding_model: str | None,
                                  local_dimensions: int | None) -> tuple[str, str | None, int | None]:
        normalized = embedding_provider.strip().lower()
        if normalized == "local_tfidf":
            return normalized, None, local_dimensions or 256
        if normalized in {"local", "local_hash", "hash"}:
            return "local_hash_v2", None, local_dimensions or 64
        if normalized in {"onnx", "onnx_local"}:
            return "onnx_local", (embedding_model or "").strip(), local_dimensions or 256
        return "sentence_transformers", (embedding_model or self._default_embedding_model).strip(), None

    def _load_embedding_provider(self, *, embedding_provider: str = "sentence_transformers",
                                 embedding_model: str | None = None, local_dimensions: int = 64) -> tuple[Any, tuple[str, str | None, int | None]]:
        normalized_provider = embedding_provider.strip().lower()
        cache_key = self._build_provider_cache_key(normalized_provider, embedding_model, local_dimensions)
        if self._provider_cache_key == cache_key and self._embedding_provider is not None:
            return self._embedding_provider, cache_key
        if self._vector_key is not None and cache_key != self._vector_key:
            # Switching providers invalidates stored vectors: re-embed fresh, never mix keys.
            self._vectors = {}
            self._vector_key = None
        try:
            provider = build_embedding_provider(
                ProviderSpec(provider=normalized_provider, model=embedding_model, dimensions=local_dimensions)
            )
        except (EmbeddingProviderError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        self._embedding_provider = provider
        self._provider_cache_key = cache_key
        return provider, cache_key

    def _ensure_semantic_index(self, provider: Any, vector_key: tuple[str, str | None, int | None]) -> None:
        """Build (or reuse) the vector index without holding the global lock during encode."""
        if self._vectors and self._vector_key == vector_key:
            return
        texts = [chunk["text"] for chunk in self._chunks]
        vectors: dict[tuple[str, int], list[float]] = {}
        if texts:
            provider.fit(texts)
            encoded = provider.encode(texts)
            if len(encoded) != len(texts):
                raise RuntimeError("embedding provider returned an unexpected number of vectors")
            dimensions = len(encoded[0]) if encoded else 0
            if dimensions <= 0 or any(
                len(vector) != dimensions or any(not math.isfinite(float(value)) for value in vector)
                for vector in encoded
            ):
                raise RuntimeError("embedding provider returned invalid or inconsistent vectors")
            vectors = {(chunk["source_id"], chunk["chunk_index"]): vec for chunk, vec in zip(self._chunks, encoded, strict=True)}
        self._vectors = vectors
        self._vector_key = vector_key
        self._store.save_vectors(vectors, embedding_key=vector_key)

    def _query_semantic(self, question: str, provider: Any, vector_key: tuple[str, str | None, int | None],
                        top_k: int, min_score: float, metadata_filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        query_vector = provider.encode([question])[0]
        expected_dimensions = next((len(vector) for vector in self._vectors.values()), None)
        if expected_dimensions is not None and len(query_vector) != expected_dimensions:
            raise RuntimeError("query embedding dimensions do not match the indexed vectors")
        source_ids = sorted({chunk["source_id"] for chunk in self._chunks}) if metadata_filter else None
        results = self._store.query_vectors(query_vector, top_k=top_k, source_ids=source_ids, embedding_key=vector_key)
        scored: list[RetrievedChunk] = []
        for source_id, chunk_index, score, chunk in results:
            if not isinstance(chunk, dict) or not chunk:
                continue
            if score <= 0.0 or score < min_score:
                continue
            if not self._matches_filter(chunk.get("metadata", {}), metadata_filter):
                continue
            scored.append(RetrievedChunk(source_id, chunk_index, chunk.get("text", ""), score, chunk.get("metadata", {})))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _query_bm25(self, question: str, top_k: int, min_score: float,
                    metadata_filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        question_tokens = [token for token in KnowledgeBase._tokenize(question) if token not in _BM25_STOP_WORDS]
        if not question_tokens:
            return []
        chunk_by_key = {(c["source_id"], c["chunk_index"]): c for c in self._chunks}
        scored: list[RetrievedChunk] = []
        for key, score in self._bm25.top_k(question_tokens, k=top_k):
            chunk = chunk_by_key.get(key)
            if chunk is None or score <= 0.0 or score < min_score:
                continue
            if not self._matches_filter(chunk["metadata"], metadata_filter):
                continue
            scored.append(RetrievedChunk(chunk["source_id"], chunk["chunk_index"], chunk["text"], score, chunk["metadata"]))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    # --------------------------------------------------------------- reranking
    def _rerank_chunks(self, question: str, chunks: list[RetrievedChunk], *, reranker: str, top_k: int) -> list[RetrievedChunk]:
        if reranker == "none":
            return chunks[:top_k]
        if reranker == "cross_encoder" and self._cross_encoder is not None:
            scores = self._cross_encoder.score_pairs(question, [chunk.text for chunk in chunks])
            reranked = [
                RetrievedChunk(chunk.source_id, chunk.chunk_index, chunk.text, float(score), chunk.metadata)
                for chunk, score in zip(chunks, scores, strict=True)
            ]
            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked[:top_k]
        question_tokens = KnowledgeBase._tokenize(question)
        if not question_tokens:
            return chunks[:top_k]
        question_terms = set(question_tokens)
        question_bigrams = set(itertools.pairwise(question_tokens))
        normalized_question = KnowledgeBase._normalize(question).lower()
        reranked = []
        for chunk in chunks:
            chunk_tokens = KnowledgeBase._tokenize(chunk.text)
            chunk_terms = set(chunk_tokens)
            chunk_bigrams = set(itertools.pairwise(chunk_tokens))
            overlap_score = len(question_terms & chunk_terms) / max(1, len(question_terms))
            bigram_score = len(question_bigrams & chunk_bigrams) / max(1, len(question_bigrams))
            exact_phrase_bonus = 1.0 if normalized_question and normalized_question in KnowledgeBase._normalize(chunk.text).lower() else 0.0
            score = (0.55 * chunk.score) + (0.3 * overlap_score) + (0.1 * bigram_score) + (0.05 * exact_phrase_bonus)
            reranked.append(RetrievedChunk(chunk.source_id, chunk.chunk_index, chunk.text, score, chunk.metadata))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    # ----------------------------------------------------------------- queries
    def query_with_trace(self, question: str, top_k: int = 5, min_score: float = 0.08, retrieval: str = "hybrid",
                         embedding_model: str | None = None, embedding_provider: str = "sentence_transformers",
                         local_dimensions: int = 64, reranker: str = "term_overlap", candidate_pool_size: int | None = None,
                         metadata_filter: dict[str, Any] | None = None, workspace_id: str | None = None
                         ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        if not question or not question.strip():
            return [], {"retrieval": retrieval, "reranker": reranker, "stages": []}
        metadata_filter = self._workspace_filter(workspace_id, metadata_filter)
        normalized_retrieval, normalized_reranker = self._validate_query_config(retrieval, reranker)
        candidate_limit = max(top_k, candidate_pool_size or top_k)

        # Multi-query retrieval (2.4): expand the question into variants when rewriting is enabled.
        queries = [question]
        if self._query_rewriter is not None and normalized_retrieval in ("tfidf", "semantic", "hybrid"):
            try:
                variants = [q.strip() for q in self._query_rewriter(question) if q and q.strip()]
            except Exception:
                variants = []
            queries = variants[:3] or [question]

        provider: Any = None
        vector_key: tuple[str, str | None, int | None] | None = None
        if normalized_retrieval in ("semantic", "hybrid"):
            provider, vector_key = self._load_embedding_provider(
                embedding_provider=embedding_provider, embedding_model=embedding_model, local_dimensions=local_dimensions)
            self._ensure_semantic_index(provider, vector_key)

        with self._lock:
            if not self._chunks:
                return [], {"retrieval": normalized_retrieval, "reranker": normalized_reranker, "stages": [],
                            "candidates_considered": 0}
            stages: list[dict[str, Any]] = []
            legs: list[list[RetrievedChunk]] = []
            if normalized_retrieval == "tfidf":
                legs = [self._query_bm25(q, candidate_limit, min_score, metadata_filter) for q in queries]
                candidates = self._merge_rankings_rrf(legs, candidate_limit)
                stages.append({"stage": "tfidf", "candidates": len(candidates)})
            elif normalized_retrieval == "semantic":
                legs = [self._query_semantic(q, provider, vector_key, candidate_limit, min_score, metadata_filter)
                        for q in queries]
                candidates = self._merge_rankings_rrf(legs, candidate_limit)
                stages.append({"stage": "semantic", "candidates": len(candidates), "embedding_provider": embedding_provider})
            else:
                bm25_legs = [self._query_bm25(q, candidate_limit, min_score, metadata_filter) for q in queries]
                semantic_legs = [self._query_semantic(q, provider, vector_key, candidate_limit, min_score, metadata_filter)
                                 for q in queries]
                legs = bm25_legs + semantic_legs
                candidates = self._merge_rankings_rrf(legs, candidate_limit)
                stages.extend([
                    {"stage": "tfidf", "candidates": sum(len(leg) for leg in bm25_legs)},
                    {"stage": "semantic", "candidates": sum(len(leg) for leg in semantic_legs),
                     "embedding_provider": embedding_provider},
                    {"stage": "rrf_fusion", "candidates": len(candidates)},
                ])
            reranked = self._rerank_chunks(question, candidates, reranker=normalized_reranker, top_k=top_k)
            if normalized_reranker != "none":
                stages.append({"stage": normalized_reranker, "candidates": len(reranked)})
            best_score = round(reranked[0].score, 4) if reranked else 0.0
            trace = {
                "retrieval": normalized_retrieval,
                "reranker": normalized_reranker,
                "candidate_pool_size": candidate_limit,
                "candidates_considered": len(candidates),
                "returned_chunks": len(reranked),
                "best_score": best_score,
                "metadata_filter": metadata_filter or {},
                "stages": stages,
            }
            return reranked, trace

    @staticmethod
    def _merge_rankings_rrf(result_sets: list[list[RetrievedChunk]], top_k: int, k: int = 60) -> list[RetrievedChunk]:
        fused_scores: dict[tuple[str, int], float] = defaultdict(float)
        canonical: dict[tuple[str, int], RetrievedChunk] = {}
        for results in result_sets:
            for rank, chunk in enumerate(results, start=1):
                key = (chunk.source_id, chunk.chunk_index)
                fused_scores[key] += 1.0 / (k + rank)
                canonical[key] = chunk
        fused = [RetrievedChunk(item.source_id, item.chunk_index, item.text, fused_scores[key], item.metadata)
                 for key, item in canonical.items()]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    def query(self, question: str, top_k: int = 5, min_score: float = 0.08, retrieval: str = "tfidf",
              embedding_model: str | None = None, embedding_provider: str = "sentence_transformers",
              local_dimensions: int = 64, reranker: str = "term_overlap", candidate_pool_size: int | None = None,
              metadata_filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        chunks, _ = self.query_with_trace(question, top_k, min_score, retrieval, embedding_model, embedding_provider,
                                          local_dimensions, reranker, candidate_pool_size, metadata_filter)
        return chunks

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            provider_name = provider_model = provider_dims = None
            if self._provider_cache_key is not None:
                provider_name, provider_model, provider_dims = self._provider_cache_key
            return {
                "documents": self._metadata_store.document_count(),
                "chunks": len(self._chunks),
                "embedding_provider": provider_name,
                "embedding_model": provider_model,
                "embedding_dimensions": provider_dims,
                "embedding_chunks_indexed": len(self._vectors),
                "default_chunk_size": self._chunk_size,
                "default_chunk_overlap": self._chunk_overlap,
                "storage": self.storage_status(),
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {"status": "ok", "storage": self.storage_status(),
                    "documents": len({c["source_id"] for c in self._chunks}), "chunks": len(self._chunks)}

    def close(self) -> None:
        """Release storage resources when the application shuts down."""
        errors: list[Exception] = []
        for resource in (self._store, self._metadata_store):
            try:
                resource.close()
            except Exception as exc:  # attempt to close both stores
                errors.append(exc)
        if errors:
            raise RuntimeError("One or more knowledge-base stores failed to close.") from errors[0]
