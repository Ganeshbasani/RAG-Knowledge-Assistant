from __future__ import annotations

from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.metadata import DocumentRecord, SQLAlchemyStore, create_metadata_store


def test_sqlalchemy_store_document_crud(tmp_path):
    store = create_metadata_store(storage_path=str(tmp_path / "index.json"))
    record = DocumentRecord(source_id="s1", filename="a.txt", file_type="txt",
                            total_chunks=3, metadata={"team": "core"})
    store.upsert_document(record)
    got = store.get_document("s1")
    assert got is not None
    assert got.source_id == "s1"
    assert got.total_chunks == 3
    assert got.metadata == {"team": "core"}
    assert store.document_count() == 1
    assert store.delete_document("s1") is True
    assert store.get_document("s1") is None
    assert store.document_count() == 0


def test_sqlalchemy_store_in_memory_when_no_path():
    store = create_metadata_store()
    assert isinstance(store, SQLAlchemyStore)
    store.upsert_document(DocumentRecord(source_id="x", total_chunks=1))
    assert store.document_count() == 1


def test_sqlalchemy_store_workspaces_feedback_audit_eval(tmp_path):
    store = create_metadata_store(storage_path=str(tmp_path / "index.json"))
    store.upsert_workspace("w1", "team-a")
    assert store.get_workspace("w1") == "team-a"
    store.add_feedback("s1", "q1", "a1", rating=5, comment="great")
    fb = store.list_feedback(source_id="s1")
    assert fb[0]["rating"] == 5
    store.add_audit_event("ingest", actor="u", target_id="s1", status="ok")
    ev = store.list_audit_events()
    assert ev[0]["action"] == "ingest"
    store.upsert_eval_result("suite-x", cases=3, hits=2, mrr=0.66, passed=True)
    r = store.list_eval_results()
    assert r[0]["suite"] == "suite-x"


def test_kb_documents_backed_by_metadata_store(tmp_path):
    kb = KnowledgeBase(storage_path=str(tmp_path / "index.json"))
    kb.ingest("doc-a", "Hello world from the alpha document.")
    kb.ingest("doc-b", "Second document about unrelated topics.")
    docs = kb.list_documents()
    assert [d["source_id"] for d in docs] == ["doc-a", "doc-b"]
    assert kb.stats()["documents"] == 2
    assert kb.remove_source("doc-a") >= 1
    assert kb.stats()["documents"] == 1
    assert kb.clear() >= 1
    assert kb.stats()["documents"] == 0
