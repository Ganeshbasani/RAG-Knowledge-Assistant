"""Chunk vector stores: the persistence seam behind ``KnowledgeBase``.

``VectorStore`` owns the durable chunk records and the embedding vectors so
that the knowledge base keeps working with in-memory retrieval state while the
backend can be JSON (default) or Qdrant for persisted ANN. The JSON backend is
the zero-dependency default; Qdrant is opt-in via ``RAG_VECTOR_STORE=qdrant``.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

STORAGE_SCHEMA_VERSION = 2
_POINT_NAMESPACE = uuid.NAMESPACE_URL


class KnowledgeBaseError(RuntimeError):
    """Base class for knowledge-base runtime failures."""


class KnowledgeBaseStorageError(KnowledgeBaseError):
    """Raised when the persistence layer cannot be loaded or written."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore(Protocol):
    """Protocol for durable chunk + vector storage backends."""

    def load_chunks(self) -> list[dict[str, Any]]: ...
    def save_chunks(self, chunks: list[dict[str, Any]]) -> None: ...
    def delete_source(self, source_id: str) -> int: ...
    def delete_all(self) -> int: ...
    def count(self) -> int: ...
    def storage_probe(self) -> bool: ...
    def close(self) -> None: ...
    def load_vectors(self) -> dict[tuple[str, int], list[float]]: ...
    def save_vectors(self, vectors: Mapping[tuple[str, int], Sequence[float]],
                     embedding_key: tuple[str, str | None, int | None] | None = None) -> None: ...
    def load_vector_key(self) -> tuple[str, str | None, int | None] | None: ...
    def query_vectors(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        source_ids: Sequence[str] | None = None,
        embedding_key: str | None = None,
    ) -> list[tuple[str, int, float, dict[str, Any]]]: ...


def dense_cosine(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Cosine similarity of two dense vectors; 0.0 when empty or dims mismatch."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _chunk_key(chunk: Mapping[str, Any]) -> tuple[str, int]:
    return (str(chunk["source_id"]), int(chunk["chunk_index"]))


class JSONVectorStore:
    """Embedded JSON file backend (the zero-dependency default).

    Preserves the historical on-disk format
    ``{"schema_version": 2, "updated_at": ..., "chunks": [...]}`` and adds an
    optional ``"vectors"`` list of ``[source_id, chunk_index, [floats]]`` so
    embeddings are persisted as well (Phase 1.5).
    """

    def __init__(self, storage_path: str | None) -> None:
        self._path = Path(storage_path).expanduser() if storage_path else None
        self._chunks: list[dict[str, Any]] = []
        self._vectors: dict[tuple[str, int], list[float]] = {}
        self._embedding_key: tuple[str, str | None, int | None] | None = None
        if self._path is not None and self._path.exists():
            self._load_from_storage()

    @property
    def kind(self) -> str:
        return "json"

    # ------------------------------------------------------------------ chunks
    def load_chunks(self) -> list[dict[str, Any]]:
        return [dict(chunk) for chunk in self._chunks]

    def save_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = [dict(chunk) for chunk in chunks]
        self._persist()

    def delete_source(self, source_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [chunk for chunk in self._chunks if chunk["source_id"] != source_id]
        removed = before - len(self._chunks)
        self._vectors = {key: value for key, value in self._vectors.items() if key[0] != source_id}
        if removed:
            self._persist()
        return removed

    def delete_all(self) -> int:
        removed = len(self._chunks)
        self._chunks = []
        self._vectors = {}
        self._persist()
        return removed

    def count(self) -> int:
        return len(self._chunks)

    def storage_probe(self) -> bool:
        if self._path is None:
            return True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            probe = self._path.with_suffix(".probe")
            probe.write_text("{}", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    def close(self) -> None:
        return None

    # ----------------------------------------------------------------- vectors
    def load_vectors(self) -> dict[tuple[str, int], list[float]]:
        return {key: list(value) for key, value in self._vectors.items()}

    def save_vectors(self, vectors: Mapping[tuple[str, int], Sequence[float]],
                     embedding_key: tuple[str, str | None, int | None] | None = None) -> None:
        self._vectors = {key: [float(v) for v in value] for key, value in vectors.items()}
        self._embedding_key = tuple(embedding_key) if embedding_key else None
        self._persist()

    def load_vector_key(self) -> tuple[str, str | None, int | None] | None:
        return self._embedding_key

    def query_vectors(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        source_ids: Sequence[str] | None = None,
        embedding_key: str | None = None,
    ) -> list[tuple[str, int, float, dict[str, Any]]]:
        allowed = set(source_ids) if source_ids else None
        chunk_by_key = {_chunk_key(chunk): chunk for chunk in self._chunks}
        scored: list[tuple[str, int, float]] = []
        for key, vector in self._vectors.items():
            source_id, chunk_index = key
            if allowed is not None and source_id not in allowed:
                continue
            score = dense_cosine(query_vector, vector)
            if score > 0.0:
                scored.append((source_id, chunk_index, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        return [(source_id, chunk_index, score, chunk_by_key.get((source_id, chunk_index), {}))
                for source_id, chunk_index, score in scored[:top_k]]

    # -------------------------------------------------------------- persistence
    def _load_from_storage(self) -> None:
        assert self._path is not None
        try:
            raw_payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseStorageError(f"failed to read persistence file: {self._path}") from exc
        if not isinstance(raw_payload, dict):
            raise KnowledgeBaseStorageError("persistence file is not in a supported format")
        chunks = raw_payload.get("chunks")
        if not isinstance(chunks, list):
            raise KnowledgeBaseStorageError("persistence file is not in a supported format")
        loaded: list[dict[str, Any]] = []
        for entry in chunks:
            if not isinstance(entry, dict):
                continue
            source_id = str(entry.get("source_id", "")).strip()
            text = str(entry.get("text", "")).strip()
            if not source_id or not text:
                continue
            try:
                chunk_index = int(entry.get("chunk_index", 0))
            except (TypeError, ValueError) as exc:
                raise KnowledgeBaseStorageError("corrupt persistence payload: chunk_index must be an int") from exc
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict):
                raise KnowledgeBaseStorageError("corrupt persistence payload: metadata must be dict")
            loaded.append({"source_id": source_id, "chunk_index": chunk_index, "text": text, "metadata": metadata})
        self._chunks = loaded

        vectors_payload = raw_payload.get("vectors")
        if isinstance(vectors_payload, list):
            loaded_vectors: dict[tuple[str, int], list[float]] = {}
            for entry in vectors_payload:
                if (isinstance(entry, (list, tuple)) and len(entry) == 3
                        and isinstance(entry[2], list)
                        and all(isinstance(v, (int, float)) for v in entry[2])):
                    loaded_vectors[(str(entry[0]), int(entry[1]))] = [float(v) for v in entry[2]]
            self._vectors = loaded_vectors
        key_payload = raw_payload.get("embedding_key")
        if isinstance(key_payload, (list, tuple)) and len(key_payload) == 3:
            provider, model, dims = key_payload
            self._embedding_key = (str(provider), model if isinstance(model, (str, type(None))) else None, dims if isinstance(dims, (int, type(None))) else None)

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "updated_at": _utc_now_iso(),
            "chunks": self._chunks,
            "vectors": [[source_id, chunk_index, vector] for (source_id, chunk_index), vector in self._vectors.items()],
            "embedding_key": list(self._embedding_key) if self._embedding_key else None,
        }
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self._path)
        except OSError as exc:
            raise KnowledgeBaseStorageError(f"failed to persist index to: {self._path}") from exc


class QdrantVectorStore:
    """Persistent vector store backed by Qdrant (vectors + chunk payloads)."""

    def __init__(self, url: str | None = None, api_key: str | None = None, collection: str = "rag-chunks") -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models as qmodels
        except ImportError as exc:
            raise KnowledgeBaseStorageError(
                "qdrant-client is required for the qdrant vector store. Install it with `pip install 'rag-knowledge-assistant[qdrant]'`."
            ) from exc
        if url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=url or "http://localhost:6333", api_key=api_key, timeout=30.0)
        self._collection = collection or "rag-chunks"
        self._qmodels = qmodels
        self._chunk_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._vector_key: tuple[str, str | None, int | None] | None = None

    @property
    def kind(self) -> str:
        return "qdrant"

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._client.close()

    def _collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def _collection_size(self) -> int | None:
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            return next(iter(vectors.values())).size
        return getattr(vectors, "size", None)

    def _ensure_collection(self, size: int = 1) -> None:
        if not self._collection_exists():
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._qmodels.VectorParams(size=size, distance=self._qmodels.Distance.COSINE),
            )

    # -------------------------------------------------------------------- chunks
    def load_chunks(self) -> list[dict[str, Any]]:
        self._chunk_cache = {}
        if not self._collection_exists():
            return []
        chunks: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=None,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                chunk = self._chunk_from_payload(record.payload or {})
                if chunk is not None:
                    chunks.append(chunk)
                    self._chunk_cache[_chunk_key(chunk)] = chunk
            if offset is None:
                break
        return chunks

    def save_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        self._ensure_collection()
        placeholder_size = self._collection_size() or 1
        points = [
            self._qmodels.PointStruct(
                id=self._point_id(chunk["source_id"], chunk["chunk_index"]),
                vector=[0.0] * placeholder_size,
                payload=self._chunk_payload(chunk),
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        for chunk in chunks:
            self._chunk_cache[_chunk_key(chunk)] = dict(chunk)

    def delete_source(self, source_id: str) -> int:
        count = self._count_points([source_id])
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._qmodels.FilterSelector(filter=self._filter([source_id])),
        )
        self._chunk_cache = {key: value for key, value in self._chunk_cache.items() if key[0] != source_id}
        return count

    def delete_all(self) -> int:
        if not self._collection_exists():
            return 0
        records, _ = self._client.scroll(
            collection_name=self._collection, limit=100000, with_payload=False, with_vectors=False
        )
        point_ids = [record.id for record in records]
        if point_ids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=self._qmodels.PointIdsList(points=point_ids),
            )
        self._chunk_cache = {}
        return len(point_ids)

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        return int(self._client.count(collection_name=self._collection).count)

    def storage_probe(self) -> bool:
        try:
            return self._collection_exists()
        except Exception:  # pragma: no cover - defensive
            return False

    # ------------------------------------------------------------------ vectors
    def load_vectors(self) -> dict[tuple[str, int], list[float]]:
        vectors: dict[tuple[str, int], list[float]] = {}
        if not self._collection_exists():
            return vectors
        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                payload = record.payload or {}
                chunk_index = payload.get("chunk_index")
                source_id = payload.get("source_id")
                vector = record.vector
                if (
                    isinstance(chunk_index, int)
                    and isinstance(source_id, str)
                    and isinstance(vector, list)
                    and vector
                    and all(isinstance(v, (int, float)) for v in vector)
                ):
                    vectors[(source_id, chunk_index)] = [float(v) for v in vector]
                    raw_key = payload.get("embedding_key")
                    if isinstance(raw_key, str):
                        try:
                            parsed_key = json.loads(raw_key)
                            if isinstance(parsed_key, list) and len(parsed_key) == 3:
                                self._vector_key = (
                                    str(parsed_key[0]),
                                    parsed_key[1] if isinstance(parsed_key[1], str) else None,
                                    parsed_key[2] if isinstance(parsed_key[2], int) else None,
                                )
                        except json.JSONDecodeError:
                            continue
            if offset is None:
                break
        return vectors

    def save_vectors(self, vectors: Mapping[tuple[str, int], Sequence[float]],
                     embedding_key: tuple[str, str | None, int | None] | None = None) -> None:
        if not vectors:
            return
        sample = next(iter(vectors.values()), None)
        if not sample:
            return
        size = len(sample)
        if not self._collection_exists():
            self._ensure_collection(size=size)
        else:
            current = self._collection_size()
            if current not in (None, size):
                # First real vectors after the size=1 placeholder: recreate the
                # collection with the real size and re-upsert chunk payloads.
                chunks = self.load_chunks()
                self._client.delete_collection(collection_name=self._collection)
                self._ensure_collection(size=size)
                if chunks:
                    self.save_chunks(chunks)
        self._vector_key = tuple(embedding_key) if embedding_key else None
        serialized_embedding_key = json.dumps(self._vector_key, separators=(",", ":")) if self._vector_key else None
        points = []
        for (source_id, chunk_index), vector in vectors.items():
            payload = self._chunk_payload(self._chunk_cache.get((source_id, chunk_index)))
            if serialized_embedding_key:
                payload["embedding_key"] = serialized_embedding_key
            points.append(
                self._qmodels.PointStruct(
                    id=self._point_id(source_id, chunk_index),
                    vector=[float(v) for v in vector],
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def load_vector_key(self) -> tuple[str, str | None, int | None] | None:
        return self._vector_key

    def query_vectors(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        source_ids: Sequence[str] | None = None,
        embedding_key: tuple[str, str | None, int | None] | None = None,
    ) -> list[tuple[str, int, float, dict[str, Any]]]:
        if not self._collection_exists():
            return []
        conditions: list[Any] = []
        if source_ids:
            conditions.append(self._qmodels.FieldCondition(
                key="source_id", match=self._qmodels.MatchAny(any=list(source_ids))
            ))
        if embedding_key:
            conditions.append(self._qmodels.FieldCondition(
                key="embedding_key",
                match=self._qmodels.MatchValue(value=json.dumps(embedding_key, separators=(",", ":")))
            ))
        query_filter = self._qmodels.Filter(must=conditions) if conditions else None
        response = self._client.query_points(
            collection_name=self._collection,
            query=[float(v) for v in query_vector],
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        out: list[tuple[str, int, float, dict[str, Any]]] = []
        for point in response.points:
            payload = point.payload or {}
            source_id = payload.get("source_id")
            chunk_index = payload.get("chunk_index")
            if isinstance(source_id, str) and isinstance(chunk_index, int):
                chunk = self._chunk_from_payload(payload) or self._chunk_cache.get((source_id, chunk_index), {})
                out.append((source_id, chunk_index, float(point.score), chunk))
        return out

    # ------------------------------------------------------------------ helpers
    def _point_id(self, source_id: str, chunk_index: int) -> Any:
        # Qdrant point ids are UUIDs; derive a deterministic one per chunk.
        return uuid.uuid5(_POINT_NAMESPACE, f"{source_id}::{chunk_index}")

    @staticmethod
    def _chunk_payload(chunk: Mapping[str, Any] | None) -> dict[str, Any]:
        if chunk is None:
            return {}
        return {
            "source_id": chunk["source_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk.get("text", ""),
            "metadata": chunk.get("metadata", {}) or {},
        }

    @staticmethod
    def _chunk_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
        source_id = payload.get("source_id")
        if not isinstance(source_id, str):
            return None
        try:
            chunk_index = int(payload.get("chunk_index", 0))
        except (TypeError, ValueError):
            return None
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return None
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {"source_id": source_id, "chunk_index": chunk_index, "text": text, "metadata": metadata}

    def _filter(self, source_ids: Sequence[str] | None) -> Any:
        conditions: list[Any] = []
        if source_ids:
            conditions.append(self._qmodels.FieldCondition(
                key="source_id", match=self._qmodels.MatchAny(any=list(source_ids))
            ))
        return self._qmodels.Filter(must=conditions) if conditions else None

    def _count_points(self, source_ids: Sequence[str] | None) -> int:
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._filter(source_ids),
            limit=100000,
            with_payload=False,
            with_vectors=False,
        )
        return len(records)


def create_vector_store(kind: str, storage_path: str | None, *, qdrant_url: str | None,
                        qdrant_api_key: str | None, qdrant_collection: str) -> VectorStore:
    normalized = (kind or "json").strip().lower()
    if normalized == "qdrant":
        return QdrantVectorStore(url=qdrant_url, api_key=qdrant_api_key, collection=qdrant_collection)
    if normalized == "json":
        return JSONVectorStore(storage_path)
    raise ValueError(f"Unknown vector store kind: {kind!r} (expected 'json' or 'qdrant')")
