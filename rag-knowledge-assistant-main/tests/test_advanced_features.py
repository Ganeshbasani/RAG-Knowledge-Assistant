from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient
from rag_assistant.knowledge_base import KnowledgeBase


def test_zero_score_results_are_filtered():
    kb = KnowledgeBase()
    kb.ingest("a", "apple banana cherry", chunk_size=10, chunk_overlap=0)
    kb.ingest("b", "dog elephant frog", chunk_size=10, chunk_overlap=0)
    assert kb.query("quantum mechanics", retrieval="tfidf", min_score=0.0) == []


def test_ingest_zero_overlap_and_replace_existing():
    kb = KnowledgeBase(chunk_size=10, chunk_overlap=2)
    first = kb.ingest_with_info("doc", "one two three four five", chunk_overlap=0)
    second = kb.ingest_with_info("doc", "updated content", chunk_overlap=0)
    assert first["replaced_existing"] is False
    assert second["replaced_existing"] is True
    assert kb.stats()["documents"] == 1
    assert kb.stats()["chunks"] == 1


def test_document_listing_and_grounded_query(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_RATE_LIMIT_REQUESTS", "100")
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    client = TestClient(api_module.app)

    ingest = client.post("/ingest", json={
        "source_id": "policy",
        "content": "Employees receive 18 days of annual leave. Leave requests require manager approval.",
        "chunk_size": 40,
        "chunk_overlap": 0,
        "metadata": {"updated_at": "2026-08-01T00:00:00+00:00", "topic": "hr"},
    })
    assert ingest.status_code == 201

    docs = client.get("/documents")
    assert docs.status_code == 200
    assert docs.json()["count"] == 1

    query = client.post("/query", json={
        "question": "How many days of annual leave do employees receive?",
        "retrieval": "hybrid",
        "embedding_provider": "local",
        "reranker": "term_overlap",
        "top_k": 2,
    })
    assert query.status_code == 200
    payload = query.json()
    assert payload["grounded"] is True
    assert payload["knowledge_gap"] is False
    assert payload["citations"]
    assert "18 days" in payload["answer"]

    unknown = client.post("/query", json={
        "question": "What is the relocation allowance?",
        "retrieval": "tfidf",
        "embedding_provider": "local",
        "reranker": "term_overlap",
    })
    assert unknown.status_code == 200
    assert unknown.json()["knowledge_gap"] is True
    assert unknown.json()["grounded"] is False


def test_file_upload_txt(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setenv("RAG_RATE_LIMIT_REQUESTS", "100")
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    client = TestClient(api_module.app)
    response = client.post("/ingest/file", files={"file": ("notes.txt", b"RAG uses retrieved context to ground answers.", "text/plain")})
    assert response.status_code == 201
    assert response.json()["file_type"] == "txt"
    assert response.json()["chunks_indexed"] >= 1
