"""Phase 2 behaviors: streaming API, sessions, query rewriting, context assembly."""

from __future__ import annotations

import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.llm import AnswerGenerator, LLMError, _rule_based_variants, extractive_fallback
from rag_assistant.sessions import SessionStore


def _fake_chunk(source_id: str, idx: int, text: str, score: float = 0.9):
    return type("C", (), {"source_id": source_id, "chunk_index": idx, "text": text, "metadata": {}, "score": score})()


# ------------------------------------------------------------------ sessions (2.3)

def test_session_store_history_ttl_and_clear():
    store = SessionStore(max_turns=3, ttl_seconds=3600)
    sid = "s1"
    store.append(sid, "user", "hello")
    store.append(sid, "assistant", "hi")
    assert [t["role"] for t in store.history(sid)] == ["user", "assistant"]
    assert store.clear(sid) == 2
    assert store.history(sid) == []
    assert store.ids() == []
    store.append(sid, "user", "again")
    store._last_seen[sid] -= 7200  # force expiry
    assert store.history(sid) == []


def test_kb_session_memory_and_history_injection():
    kb = KnowledgeBase()
    kb.ingest("doc", "Remote work policy: three days per week.")
    assert kb.augment_with_history("", "hello") == "hello"
    kb.record_turn("s1", "user", "how many days?")
    augmented = kb.augment_with_history("s1", "what else?")
    assert "how many days?" in augmented
    assert "what else?" in augmented
    assert kb.session_ids() == ["s1"]
    assert kb.clear_session("s1") == 1


# ----------------------------------------------------------------- query rewriting (2.4)

def test_rule_based_variants():
    variants = _rule_based_variants("What is the remote work policy and how many days are allowed")
    assert len(variants) >= 1
    assert all(isinstance(v, str) and v for v in variants)


def test_answer_generator_rewrite_fallback():
    generator = AnswerGenerator(provider="extractive")
    variants = generator.rewrite_query("budget and manager sign-off")
    assert variants[:2] == ["budget", "manager sign-off"]


def test_extractive_fallback_summarizes_document_without_contact_details():
    answer = extractive_fallback("Exaplian the doc", [{
        "text": "BASANI GANESH Email: ganesh@example.com. PROFESSIONAL SUMMARY "
                "2026 Computer Science graduate with experience building Python backend applications. "
                "Skilled in FastAPI and SQL. SKILLS Python, FastAPI, SQL",
    }])

    assert "2026 Computer Science graduate" in answer
    assert "Skilled in FastAPI and SQL" in answer
    assert "ganesh@example.com" not in answer


def test_extractive_fallback_maps_qualification_to_education_terms():
    answer = extractive_fallback("What is the qualification?", [{
        "text": "Education: B.Tech in Computer Science and Engineering, CGPA 8.03/10.",
    }])

    assert "B.Tech in Computer Science" in answer


def test_kb_multiquery_retrieves_variant_terms():
    kb = KnowledgeBase()
    kb.ingest("doc-a", "The remote work policy allows three days a week.")
    kb.ingest("doc-b", "The performance review process is annual.")
    chunks, trace = kb.query_with_trace("remote work and performance review", retrieval="tfidf", reranker="none")
    assert chunks
    assert any(stage["stage"] == "tfidf" for stage in trace["stages"])


# ----------------------------------------------------------------- context assembly (2.5)

def test_strip_page_markers():
    from rag_assistant.api import _strip_page_markers

    result = _strip_page_markers("alpha [PAGE 2] and more")
    assert "PAGE" not in result
    assert "and more" in result


def test_dedup_context_collapses_near_duplicates():
    from rag_assistant.api import _dedup_context

    chunks = [_fake_chunk("a", 0, "alpha beta gamma delta"), _fake_chunk("a", 1, "alpha beta gamma delta")]
    assert len(_dedup_context(chunks)) == 1


def test_trim_context_respects_token_budget():
    from rag_assistant.api import _trim_context

    chunks = [_fake_chunk("a", i, "a" * 20) for i in range(10)]  # 5 tokens each
    kept = _trim_context(chunks, max_tokens=12)
    assert len(kept) == 2
    assert _trim_context(chunks, max_tokens=0) is chunks


def test_query_request_context_and_stream_defaults():
    from rag_assistant.models import QueryRequest

    request = QueryRequest(question="q")
    assert request.context_max_tokens == 0
    assert request.stream is False


# ------------------------------------------------------------------ streaming + sessions API (2.1, 2.3)

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


def test_api_streaming_query_events(api_client):
    client, _ = api_client
    client.post("/ingest", json={"source_id": "doc", "content": "The remote work policy allows three days a week."})
    response = client.post("/query", json={"question": "How many remote days?", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[len("data: "):]) for line in response.text.split("\n\n")
              if line.startswith("data: ")]
    assert events
    assert all(event["type"] in {"delta", "replace", "done"} for event in events)
    done = [event for event in events if event["type"] == "done"]
    assert done and done[0]["response"]["count"] >= 1
    assert done[0]["response"]["answer"]


def test_api_replaces_partial_stream_after_provider_failure(api_client, monkeypatch):
    client, api_module = api_client
    client.post("/ingest", json={
        "source_id": "remote-policy",
        "content": "The remote work policy allows three days every week.",
    })

    def interrupted_stream(*_args, **_kwargs):
        yield "A partial model answer"
        raise LLMError("Provider stream interrupted.", category="provider_stream_interrupted")

    api_module.answer_generator._adapter = type("FakeAdapter", (), {"model": "test-model"})()
    monkeypatch.setattr(api_module.answer_generator, "generate_stream", interrupted_stream)
    response = client.post("/query", json={"question": "How many remote days each week?", "stream": True})
    assert response.status_code == 200
    events = [json.loads(line[len("data: "):]) for line in response.text.split("\n\n")
              if line.startswith("data: ")]
    assert any(event["type"] == "delta" for event in events)
    replacement = next(event for event in events if event["type"] == "replace")
    done = next(event for event in events if event["type"] == "done")
    assert "partial model answer" not in replacement["text"]
    assert replacement["text"] == done["response"]["answer"]
    assert done["response"]["generation"]["error_category"] == "provider_stream_interrupted"


def test_api_completes_successful_provider_stream(api_client, monkeypatch):
    client, api_module = api_client
    client.post("/ingest", json={"source_id": "remote-policy", "content": "Remote work is allowed three days per week."})

    def complete_stream(*_args, **_kwargs):
        yield "Remote work is allowed three days per week. [1]"

    api_module.answer_generator._adapter = type("FakeAdapter", (), {"model": "test-model"})()
    monkeypatch.setattr(api_module.answer_generator, "generate_stream", complete_stream)
    response = client.post("/query", json={"question": "How many days per week?", "stream": True})
    events = [json.loads(line[len("data: "):]) for line in response.text.split("\n\n")
              if line.startswith("data: ")]
    done = next(event["response"] for event in events if event["type"] == "done")
    assert done["answer"] == "Remote work is allowed three days per week. [1]"
    assert done["generation"]["used_llm"] is True


def test_api_sessions_endpoints(api_client):
    client, _ = api_client
    client.post("/ingest", json={"source_id": "doc", "content": "Alpha content about apples."})
    response = client.post("/query", json={"question": "What is alpha?", "session_id": "s-1"})
    assert response.status_code == 200
    listing = client.get("/sessions")
    assert listing.status_code == 200
    assert "s-1" in listing.json()["sessions"]
    cleared = client.delete("/sessions/s-1")
    assert cleared.status_code == 200
    assert cleared.json()["removed_turns"] == 2
    assert client.delete("/sessions/s-1").json()["removed_turns"] == 0
    assert client.delete("/sessions/ghost").json()["removed_turns"] == 0
