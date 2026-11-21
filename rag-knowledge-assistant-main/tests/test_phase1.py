"""Phase 1 behaviors: store-owned vectors, BM25 lexical, async ingestion."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.knowledge_base import KnowledgeBase

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_kb_semantic_vectors_persist_across_reload(tmp_path):
    path = tmp_path / "index.json"
    kb = KnowledgeBase(storage_path=str(path))
    kb.ingest("doc", "alpha beta gamma delta epsilon")
    kb.query_with_trace("alpha", retrieval="semantic", embedding_provider="local", reranker="none")
    assert kb.stats()["embedding_chunks_indexed"] >= 1
    assert kb._store.load_vector_key() is not None
    reloaded = KnowledgeBase(storage_path=str(path))
    assert reloaded.stats()["embedding_chunks_indexed"] >= 1


def test_kb_provider_switch_invalidates_vectors():
    kb = KnowledgeBase()
    kb.ingest("doc", "some content here for provider switching")
    kb.query_with_trace("content", retrieval="semantic", embedding_provider="local", reranker="none")
    key1 = kb._vector_key
    kb.query_with_trace("content", retrieval="semantic", embedding_provider="local_hash",
                        local_dimensions=8, reranker="none")
    assert kb._vector_key != key1


def test_kb_tfidf_mode_uses_bm25():
    kb = KnowledgeBase()
    kb.ingest("doc", "the quick brown fox jumps over the lazy dog")
    chunks, trace = kb.query_with_trace("quick brown fox", retrieval="tfidf", reranker="none")
    assert chunks and chunks[0].source_id == "doc"
    assert any(stage["stage"] == "tfidf" for stage in trace["stages"])


def test_kb_ingest_and_query_end_to_end():
    kb = KnowledgeBase()
    kb.ingest("doc", "Remote work policy allows three days per week.")
    chunks, _ = kb.query_with_trace("How many remote days?", retrieval="tfidf", reranker="none")
    assert chunks


def test_kb_reranker_cross_encoder_config():
    pytest.importorskip("sentence_transformers")
    kb = KnowledgeBase(reranker="cross_encoder")
    assert kb._reranker_name == "cross_encoder"


def test_cross_encoder_reranker_scores_pairs():
    pytest.importorskip("sentence_transformers")
    from rag_assistant.rerankers import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scores = reranker.score_pairs(
        "remote work policy",
        ["Remote work policy: three days a week.", "Unrelated content."],
    )
    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)


def test_kb_rejects_unknown_reranker():
    kb = KnowledgeBase()
    with pytest.raises(ValueError, match="reranker"):
        kb.query_with_trace("q", retrieval="tfidf", reranker="bogus")


# ---------------------------------------------------------------- migration


def test_migrate_json_to_qdrant(tmp_path):
    pytest.importorskip("qdrant_client")
    import importlib.util

    from rag_assistant.stores import JSONVectorStore

    storage_path = str(tmp_path / "index.json")
    json_store = JSONVectorStore(storage_path)
    json_store.save_chunks([
        {"source_id": "a", "chunk_index": 0, "text": "alpha text about apples", "metadata": {}},
        {"source_id": "a", "chunk_index": 1, "text": "more apples here", "metadata": {}},
    ])
    json_store.save_vectors(
        {("a", 0): [0.5, 0.5, 0.5, 0.5], ("a", 1): [0.5, 0.5, 0.5, 0.5]},
        embedding_key=("sentence_transformers", "m", 4),
    )

    script_path = _REPO_ROOT / "scripts" / "migrate_json_to_qdrant.py"
    spec = importlib.util.spec_from_file_location("migrate_json_to_qdrant", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.migrate(storage_path, ":memory:", "test-migrate", dry_run=True) == 0
    assert module.migrate(storage_path, ":memory:", "test-migrate") == 0


# ---------------------------------------------------------------- async ingest


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    with TestClient(api_module.app) as client:
        yield client, api_module


def test_api_async_ingest_flow(api_client):
    client, _ = api_client
    response = client.post("/ingest/async", json={"source_id": "async-doc", "content": "Async ingestion works fine."})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get(f"/ingest/status/{job_id}")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "indexed"
    assert payload["result"]["source_id"] == "async-doc"
    assert client.get("/ingest/status/nope").status_code == 404
