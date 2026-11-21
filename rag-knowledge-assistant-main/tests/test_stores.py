from __future__ import annotations

import json

import pytest

from rag_assistant.stores import (
    JSONVectorStore,
    KnowledgeBaseStorageError,
    QdrantVectorStore,
    dense_cosine,
)


def _chunk(source_id: str, idx: int, text: str) -> dict:
    return {"source_id": source_id, "chunk_index": idx, "text": text, "metadata": {"source": source_id}}


# ---------------------------------------------------------------- JSON backend


def test_json_store_round_trip(tmp_path):
    path = tmp_path / "index.json"
    store = JSONVectorStore(str(path))
    assert store.count() == 0
    assert store.storage_probe() is True
    store.save_chunks([_chunk("a", 0, "alpha"), _chunk("a", 1, "alphabet")])
    assert store.count() == 2
    reloaded = JSONVectorStore(str(path))
    assert {c["source_id"] for c in reloaded.load_chunks()} == {"a"}
    assert reloaded.count() == 2
    assert {c["text"] for c in reloaded.load_chunks()} == {"alpha", "alphabet"}


def test_json_store_legacy_format_without_vectors(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({"schema_version": 2, "chunks": [{"source_id": "a", "chunk_index": 0, "text": "alpha", "metadata": {}}]}),
        encoding="utf-8",
    )
    store = JSONVectorStore(str(path))
    assert store.count() == 1
    assert store.load_vectors() == {}


def test_json_store_delete_source(tmp_path):
    store = JSONVectorStore(None)
    store.save_chunks([_chunk("a", 0, "alpha"), _chunk("b", 0, "beta")])
    assert store.delete_source("a") == 1
    assert [c["source_id"] for c in store.load_chunks()] == ["b"]


def test_json_store_delete_all(tmp_path):
    store = JSONVectorStore(None)
    store.save_chunks([_chunk("a", 0, "alpha"), _chunk("b", 0, "beta")])
    assert store.delete_all() == 2
    assert store.count() == 0


def test_json_store_corrupt_file_raises(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(KnowledgeBaseStorageError):
        JSONVectorStore(str(path))


def test_json_store_vectors_persist_with_key(tmp_path):
    path = tmp_path / "index.json"
    store = JSONVectorStore(str(path))
    store.save_chunks([_chunk("a", 0, "alpha")])
    store.save_vectors({("a", 0): [0.1, 0.2, 0.3]}, embedding_key=("sentence_transformers", "model-x", None))
    reloaded = JSONVectorStore(str(path))
    assert reloaded.load_vector_key() == ("sentence_transformers", "model-x", None)
    assert reloaded.load_vectors() == {("a", 0): [0.1, 0.2, 0.3]}


def test_json_store_query_vectors_ranks_by_cosine():
    store = JSONVectorStore(None)
    store.save_chunks([_chunk("a", 0, "x"), _chunk("b", 0, "y")])
    store.save_vectors({("a", 0): [1.0, 0.0], ("b", 0): [0.0, 1.0]})
    results = store.query_vectors([1.0, 0.0], top_k=5)
    assert results[0][:2] == ("a", 0)
    assert dense_cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


# ---------------------------------------------------------------- Qdrant backend


def test_qdrant_store_round_trip():
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(url=":memory:", collection="test-round-trip")
    assert store.count() == 0
    store.save_chunks([_chunk("a", 0, "alpha"), _chunk("a", 1, "alphabet"), _chunk("b", 0, "beta")])
    assert store.count() == 3
    loaded = store.load_chunks()
    assert {(c["source_id"], c["chunk_index"]) for c in loaded} == {("a", 0), ("a", 1), ("b", 0)}
    assert {c["text"] for c in loaded} == {"alpha", "alphabet", "beta"}


def test_qdrant_store_save_and_load_vectors():
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(url=":memory:", collection="test-vectors")
    store.save_chunks([_chunk("a", 0, "alpha")])
    # Qdrant normalizes vectors for COSINE collections; use a unit vector.
    store.save_vectors({("a", 0): [1.0, 0.0, 0.0]}, embedding_key=("sentence_transformers", "m", 384))
    assert store.load_vectors() == {("a", 0): [1.0, 0.0, 0.0]}
    assert store.load_vector_key() == ("sentence_transformers", "m", 384)


def test_qdrant_store_query_vectors_ranks_by_cosine():
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(url=":memory:", collection="test-query")
    store.save_chunks([_chunk("a", 0, "alpha"), _chunk("b", 0, "beta")])
    store.save_vectors({("a", 0): [1.0, 0.0], ("b", 0): [0.0, 1.0]})
    results = store.query_vectors([1.0, 0.0], top_k=5)
    assert results[0][0] == "a"
    assert results[0][3]["text"] == "alpha"


def test_qdrant_store_delete_source_and_count():
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(url=":memory:", collection="test-delete")
    store.save_chunks([_chunk("a", 0, "alpha"), _chunk("a", 1, "alphabet"), _chunk("b", 0, "beta")])
    assert store.delete_source("a") == 2
    assert store.count() == 1
    assert [c["source_id"] for c in store.load_chunks()] == ["b"]
