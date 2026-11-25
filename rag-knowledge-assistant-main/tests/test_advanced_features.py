from __future__ import annotations

import importlib
import sys
from io import BytesIO

from fastapi.testclient import TestClient

from rag_assistant.knowledge_base import KnowledgeBase


def test_knowledge_base_close_releases_both_stores(monkeypatch):
    kb = KnowledgeBase()
    closed = []
    monkeypatch.setattr(kb._store, "close", lambda: closed.append("vectors"))
    monkeypatch.setattr(kb._metadata_store, "close", lambda: closed.append("metadata"))

    kb.close()

    assert closed == ["vectors", "metadata"]


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
    assert payload["confidence"] is None
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
    library = client.get("/documents")
    assert library.status_code == 200
    assert library.json()["count"] == 1
    assert library.json()["documents"][0]["chunks"] == response.json()["chunks_indexed"]


def test_readable_pdf_upload_index_retrieval_and_citation(monkeypatch, tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RAG_ENV", "development")
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.delenv("RAG_JWT_SECRET", raising=False)
    monkeypatch.delenv("RAG_UI_PASSWORD", raising=False)
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    client = TestClient(api_module.app)

    pdf_buffer = BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (Atlas Demo Orchard: the north orchard has 48 apple trees.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(pdf_buffer)

    upload = client.post("/ingest/file", files={"file": ("orchard-report.pdf", pdf_buffer.getvalue(), "application/pdf")})
    assert upload.status_code == 201, upload.text
    uploaded = upload.json()
    assert uploaded["chunks_indexed"] >= 1
    documents = client.get("/documents").json()
    assert documents["count"] == 1
    document = documents["documents"][0]
    assert document["metadata"]["page_count"] == 1
    assert document["chunks"] == uploaded["chunks_indexed"]

    response = client.post("/query", json={
        "question": "How many apple trees are in the north orchard?",
        "retrieval": "hybrid",
        "embedding_provider": "local",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert "48 apple trees" in result["answer"]
    assert result["citations"]
    citation = result["citations"][0]
    assert citation["source_id"] == uploaded["source_id"]
    assert any("48 apple trees" in chunk["text"] for chunk in result["context"])
    assert client.get("/stats").json()["embedding_chunks_indexed"] >= uploaded["chunks_indexed"]
    unrelated = client.post("/query", json={
        "question": "What is the lunar landing budget?",
        "retrieval": "hybrid",
        "embedding_provider": "local",
    })
    assert unrelated.status_code == 200
    assert unrelated.json()["knowledge_gap"] is True
    assert unrelated.json()["citations"] == []
