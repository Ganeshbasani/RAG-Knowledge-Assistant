"""Regression tests for the Phase 0 truthfulness/safety fixes.

Covers: the 413/400 error contract (declared and streamed bodies), /clear
confirmation, DELETE 404 on unknown sources, atomic bulk ingest, /ingest/file
query-parameter validation, orphaned-file cleanup, the new default embedder,
production fail-fast, and the metadata size cap.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.models import EvalCase, IngestRequest, QueryRequest


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    """Import a fresh api module against temp storage and return (client, module)."""
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")

    with TestClient(api_module.app) as client:
        yield client, api_module


# --------------------------------------------------------------------------
# 413 / 400 contract for declared bodies
# --------------------------------------------------------------------------

def test_oversized_content_length_returns_413(api_client):
    client, _ = api_client
    response = client.post(
        "/ingest",
        json={"source_id": "too-big", "content": "x" * 100},
        headers={"content-length": "99999999"},
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


def test_invalid_content_length_returns_400(api_client):
    client, _ = api_client
    response = client.post(
        "/ingest",
        json={"source_id": "bad-length", "content": "x"},
        headers={"content-length": "not-a-number"},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_request"


# --------------------------------------------------------------------------
# Streamed body limit (chunked transfer has no Content-Length)
# --------------------------------------------------------------------------

@pytest.fixture
def api_with_small_body_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RAG_MAX_REQUEST_BYTES", "200")
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    return api_module


def test_streamed_body_over_limit_returns_413(api_with_small_body_limit):
    client = TestClient(api_with_small_body_limit.app)

    def chunked_body():
        yield b"a" * 250
        yield b"b" * 250

    response = client.post("/ingest", content=chunked_body(), headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


# --------------------------------------------------------------------------
# /clear confirmation + DELETE 404
# --------------------------------------------------------------------------

def test_clear_requires_confirmation_header(api_client):
    client, _ = api_client
    client.post("/ingest", json={"source_id": "doc", "content": "some content"})

    unconfirmed = client.delete("/clear")
    assert unconfirmed.status_code == 400

    confirmed = client.delete("/clear", headers={"X-Clear-Confirm": "DELETE"})
    assert confirmed.status_code == 200


def test_delete_unknown_source_returns_404(api_client):
    client, _ = api_client
    response = client.delete("/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "not_found"


# --------------------------------------------------------------------------
# Atomic bulk ingest
# --------------------------------------------------------------------------

def test_bulk_ingest_is_atomic_on_failure(api_client):
    client, _ = api_client
    response = client.post(
        "/ingest/bulk",
        json={"documents": [
            {"source_id": "bulk-a", "content": "alpha"},
            {"source_id": "bulk-b", "content": "beta"},
            {"source_id": "bulk-bad", "content": " "},
        ]},
    )
    assert response.status_code == 400
    stats = client.get("/stats").json()
    assert stats["chunks"] == 0


def test_bulk_ingest_success_indexes_all_documents(api_client):
    client, _ = api_client
    response = client.post(
        "/ingest/bulk",
        json={"documents": [
            {"source_id": "bulk-a", "content": "alpha"},
            {"source_id": "bulk-b", "content": "beta"},
        ]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"] == 2
    assert payload["chunks_indexed"] == stats_chunks(client)


def stats_chunks(client):
    return client.get("/stats").json()["chunks"]


def test_knowledge_base_ingest_many_is_atomic():
    kb = KnowledgeBase()
    with pytest.raises(ValueError):
        kb.ingest_many([
            {"source_id": "a", "content": "hello world"},
            {"source_id": "b", "content": "   "},
        ])
    assert kb.stats()["chunks"] == 0

    total = kb.ingest_many([
        {"source_id": "a", "content": "hello world"},
        {"source_id": "b", "content": "foo bar"},
    ])
    assert total == 2
    assert kb.stats()["documents"] == 2


# --------------------------------------------------------------------------
# /ingest/file validation + no orphaned artifacts
# --------------------------------------------------------------------------

def test_file_upload_source_id_max_length_enforced(api_client):
    client, _ = api_client
    response = client.post(
        "/ingest/file",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        params={"source_id": "A" * 500},
    )
    assert response.status_code == 422


def test_file_upload_chunk_bounds_enforced(api_client):
    client, _ = api_client
    base = {"files": {"file": ("notes.txt", b"hello world", "text/plain")}}
    assert client.post("/ingest/file", params={"chunk_size": "-5"}, **base).status_code == 422
    assert client.post("/ingest/file", params={"chunk_size": "5000"}, **base).status_code == 422
    assert client.post("/ingest/file", params={"chunk_overlap": "8000"}, **base).status_code == 422


def test_file_upload_failure_writes_no_orphan_artifact(api_client, tmp_path):
    client, _ = api_client
    response = client.post("/ingest/file", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 400
    upload_dir = tmp_path / "uploads"
    assert not upload_dir.exists() or list(upload_dir.iterdir()) == []


# --------------------------------------------------------------------------
# Default embedder / metadata cap
# --------------------------------------------------------------------------

def test_query_and_eval_default_provider_is_local():
    assert QueryRequest(question="q?").embedding_provider == "local"
    assert EvalCase(question="q?", expected_source_ids=["a"]).embedding_provider == "local"


def test_metadata_size_cap_enforced():
    IngestRequest(source_id="s", content="c", metadata={"small": "ok"})
    with pytest.raises(ValidationError):
        IngestRequest(source_id="s", content="c", metadata={"blob": "x" * (70 * 1024)})


# --------------------------------------------------------------------------
# Production fail-fast
# --------------------------------------------------------------------------

def test_production_mode_without_api_key_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_ENV", "production")
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    with pytest.raises(RuntimeError, match="RAG_ENV=production"):
        importlib.import_module("rag_assistant.api")

    # Restore a healthy module for subsequent tests.
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    monkeypatch.setenv("RAG_ENV", "development")
    importlib.import_module("rag_assistant.api")
