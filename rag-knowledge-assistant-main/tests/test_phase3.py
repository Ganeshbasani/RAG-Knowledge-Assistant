"""Phase 3 behaviors: injection hardening, citation verification, eval harness, feedback, audit log."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api import _grounded_status, _verify_citations
from rag_assistant.evals import abstention_accuracy, citation_precision, load_cases, recall_at_k
from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.llm import _build_messages
from rag_assistant.models import Citation, ContextChunk

_FIXTURES = Path(__file__).parent / "fixtures" / "eval"


# ------------------------------------------------------------------ 3.1 injection hardening

def test_injection_hardening_wraps_untrusted_content():
    system, user = _build_messages(
        "What is RAG?",
        [{"text": "evil doc", "source_id": "d", "chunk_index": 0, "metadata": {}}],
        [{"source_id": "d", "chunk_index": 0, "label": "d"}],
    )
    assert "<<<UNTRUSTED QUESTION>>>" in user
    assert "<<<END UNTRUSTED QUESTION>>>" in user
    assert "<<<UNTRUSTED CONTEXT #1" in user
    assert "<<<END UNTRUSTED CONTEXT #1" in user
    assert "SECURITY" in system


# ------------------------------------------------------------------ 3.2 citation verification

def _make_citations():
    return [
        Citation(citation_id=1, source_id="a", chunk_index=0, label="a"),
        Citation(citation_id=2, source_id="b", chunk_index=0, label="b"),
    ]


def _make_context():
    return [
        ContextChunk(source_id="a", chunk_index=0, text="alpha apples", score=0.9, metadata={}),
        ContextChunk(source_id="b", chunk_index=0, text="beta bananas", score=0.8, metadata={}),
    ]


def test_verify_citations_reports_missing_and_overlap():
    report = _verify_citations("[1] Apples are red.", _make_citations(), _make_context())
    assert report[0]["cited"] is True
    assert report[0]["chunk_token_overlap"] > 0
    assert report[1]["cited"] is False


def test_grounded_status_extractive_true_llm_verified():
    citations, context = _make_citations(), _make_context()
    grounded, verification = _grounded_status("[1] Apples are red. [2] Bananas are yellow.", citations, context,
                                              used_llm=True)
    assert grounded is True
    assert all(item["cited"] and item["chunk_token_overlap"] > 0 for item in verification)
    grounded_missing, _ = _grounded_status("The answer with no citations here.", citations, context, used_llm=True)
    assert grounded_missing is False
    grounded_unsupported, verification_unsupported = _grounded_status("[1] Something else entirely.", citations, context,
                                                                     used_llm=True)
    assert grounded_unsupported is False
    assert verification_unsupported[0]["chunk_token_overlap"] == 0.0


# ------------------------------------------------------------------ 3.3+3.4 eval datasets + harness

def test_eval_datasets_load():
    cases = load_cases(str(_FIXTURES / "retrieval_cases.jsonl"))
    assert len(cases) >= 5
    assert all(case.get("expected_source_ids") for case in cases)
    adversarial = load_cases(str(_FIXTURES / "adversarial_cases.jsonl"))
    assert len(adversarial) >= 4
    answerable = load_cases(str(_FIXTURES / "answerable_cases.jsonl"))
    assert all("must_contain" in case for case in answerable)
    unanswerable = load_cases(str(_FIXTURES / "unanswerable_cases.jsonl"))
    assert all(not case.get("expected_source_ids") for case in unanswerable)
    conflicts = load_cases(str(_FIXTURES / "conflict_cases.jsonl"))
    assert len(conflicts) == 2


def test_eval_metrics_on_synthetic_results():
    results = [
        {"expected_source_ids": ["a", "b"], "returned_source_ids": ["a", "b", "c"]},
        {"expected_source_ids": [], "returned_source_ids": []},
        {"expected_source_ids": [], "returned_source_ids": ["x"]},
    ]
    assert recall_at_k(results, k=2) == 1.0
    assert citation_precision(results) == 1.0
    assert abstention_accuracy(results) == 0.5
    assert abstention_accuracy([{"expected_source_ids": [], "returned_source_ids": []}]) == 1.0


def test_kb_sessions_and_feedback_integration():
    kb = KnowledgeBase()
    kb.ingest("doc", "Alpha content about apples.")
    kb.add_feedback("doc", "q?", "a?", rating=5, comment="nice")
    feedback = kb.list_feedback(source_id="doc")
    assert feedback and feedback[0]["rating"] == 5


# ------------------------------------------------------------------ 3.9 audit log + CSP

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


def test_audit_log_records_destructive_ops(api_client):
    client, api_module = api_client
    client.post("/ingest", json={"source_id": "doc", "content": "Alpha about apples."})
    client.delete("/documents/doc")
    events = api_module.kb._metadata_store.list_audit_events(limit=5)
    assert any(e["action"] == "remove_source" and e["target_id"] == "doc" for e in events)
    api_module.kb.clear()
    events = api_module.kb._metadata_store.list_audit_events(limit=10)
    assert any(e["action"] == "clear_all" for e in events)


def test_csp_header_set_on_responses(api_client):
    client, _ = api_client
    response = client.get("/health")
    assert response.headers.get("content-security-policy")


# ------------------------------------------------------------------ 3.8 feedback API

def test_feedback_endpoints(api_client):
    client, _ = api_client
    client.post("/ingest", json={"source_id": "doc", "content": "Alpha about apples."})
    query = client.post("/query", json={"question": "What is alpha?", "session_id": "s1"})
    assert query.status_code == 200
    feedback = client.post("/feedback", json={
        "source_id": "doc", "question": "What is alpha?",
        "answer": query.json()["answer"], "rating": 5, "comment": "helpful",
    })
    assert feedback.status_code == 200
    assert feedback.json()["feedback_id"] >= 1
    listing = client.get("/feedback", params={"source_id": "doc"})
    assert listing.status_code == 200
    assert listing.json()["count"] >= 1
    invalid = client.post("/feedback", json={"source_id": "doc", "question": "?", "answer": "a", "rating": 6})
    assert invalid.status_code == 422
