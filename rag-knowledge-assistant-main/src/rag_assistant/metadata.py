"""Structured metadata store: SQLAlchemy models over SQLite (Postgres-ready).

Backs ``KnowledgeBase.list_documents()`` / ``stats()`` and powers the
workspace/feedback/audit features for the production architecture. Default
database: SQLite beside the chunk store (``rag-metadata.db``); set
``RAG_DB_URL`` (or pass ``db_url=``) for PostgreSQL. When no ``storage_path``
is configured, an in-memory SQLite database is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import JSON, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetadataStore(Protocol):
    """Protocol for structured document/workspace/feedback/audit stores."""

    def upsert_document(self, record: DocumentRecord) -> None: ...
    def get_document(self, source_id: str) -> DocumentRecord | None: ...
    def list_documents(self) -> list[DocumentRecord]: ...
    def document_count(self) -> int: ...
    def delete_document(self, source_id: str) -> bool: ...
    def delete_all_documents(self) -> int: ...
    def upsert_workspace(self, workspace_id: str, name: str) -> None: ...
    def get_workspace(self, workspace_id: str) -> str | None: ...
    def list_workspaces(self) -> list[tuple[str, str]]: ...
    def add_feedback(self, source_id: str, question: str, answer: str, rating: int | None = None,
                     comment: str | None = None) -> None: ...
    def list_feedback(self, source_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]: ...
    def add_audit_event(self, action: str, actor: str | None = None, target_id: str | None = None,
                        status: str | None = None, details: dict[str, Any] | None = None) -> None: ...
    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]: ...
    def upsert_eval_result(self, suite: str, cases: int, hits: int, mrr: float, passed: bool) -> None: ...
    def list_eval_results(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


# Keys stripped from chunk metadata when building DocumentRecord.metadata summaries.
SYSTEM_METADATA_KEYS = frozenset({
    "filename", "file_type", "content_hash", "ingested_at", "updated_at", "token_count", "chunking_strategy",
})


class DocumentRecord:
    """One document row (source of truth for ``/documents``)."""

    def __init__(self, source_id: str, workspace_id: str | None = None, filename: str | None = None,
                 file_type: str | None = None, content_hash: str | None = None, total_chunks: int = 0,
                 ingested_at: str | None = None, updated_at: str | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
        self.source_id = source_id
        self.workspace_id = workspace_id
        self.filename = filename
        self.file_type = file_type
        self.content_hash = content_hash
        self.total_chunks = total_chunks
        self.ingested_at = ingested_at or _utc_now_iso()
        self.updated_at = updated_at or self.ingested_at
        self.metadata = metadata or {}

    @classmethod
    def from_chunk_metadata(cls, source_id: str, total_chunks: int, chunk_metadata: dict[str, Any]) -> DocumentRecord:
        def text(key: str) -> str | None:
            value = chunk_metadata.get(key)
            return str(value) if value is not None else None

        return cls(
            source_id=source_id,
            workspace_id=text("workspace_id"),
            filename=text("filename"),
            file_type=text("file_type"),
            content_hash=text("content_hash"),
            total_chunks=total_chunks,
            ingested_at=text("ingested_at"),
            updated_at=text("updated_at") or text("ingested_at"),
            metadata={k: v for k, v in chunk_metadata.items() if k not in SYSTEM_METADATA_KEYS},
        )

    def as_summary(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunks": self.total_chunks,
            "filename": self.filename,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class _Base(DeclarativeBase):
    pass


class _Document(_Base):
    __tablename__ = "documents"
    source_id: Mapped[str] = mapped_column(primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(default=None, index=True)
    filename: Mapped[str | None] = mapped_column(default=None)
    file_type: Mapped[str | None] = mapped_column(default=None)
    content_hash: Mapped[str | None] = mapped_column(default=None)
    total_chunks: Mapped[int] = mapped_column(default=0)
    ingested_at: Mapped[str] = mapped_column(default=_utc_now_iso)
    updated_at: Mapped[str] = mapped_column(default=_utc_now_iso)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    @classmethod
    def record(cls) -> DocumentRecord:
        raise NotImplementedError


class _Workspace(_Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(default="")
    created_at: Mapped[str] = mapped_column(default=_utc_now_iso)


class _Feedback(_Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column()
    question: Mapped[str] = mapped_column()
    answer: Mapped[str] = mapped_column()
    rating: Mapped[int | None] = mapped_column(default=None)
    comment: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[str] = mapped_column(default=_utc_now_iso)


class _AuditEvent(_Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    actor: Mapped[str | None] = mapped_column(default=None)
    action: Mapped[str] = mapped_column()
    target_id: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[str | None] = mapped_column(default=None)
    details_json: Mapped[dict[str, Any]] = mapped_column("details", JSON, default=dict)
    created_at: Mapped[str] = mapped_column(default=_utc_now_iso)


class _EvalResult(_Base):
    __tablename__ = "eval_results"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column()
    cases: Mapped[int] = mapped_column(default=0)
    hits: Mapped[int] = mapped_column(default=0)
    mrr: Mapped[float] = mapped_column(default=0.0)
    passed: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[str] = mapped_column(default=_utc_now_iso)


class SQLAlchemyStore:
    """Structured metadata store backed by SQLAlchemy (SQLite by default)."""

    def __init__(self, db_url: str) -> None:
        from sqlalchemy.pool import StaticPool

        kwargs: dict[str, Any] = {}
        if db_url == "sqlite://":
            # In-memory SQLite must share one connection across threads.
            kwargs["poolclass"] = StaticPool
        self._engine = create_engine(db_url, connect_args=self._connect_args(db_url), **kwargs)
        _Base.metadata.create_all(self._engine)
        self._session = sessionmaker(bind=self._engine, expire_on_commit=False)

    @staticmethod
    def _connect_args(db_url: str) -> dict[str, Any]:
        if db_url.startswith("sqlite"):
            return {"check_same_thread": False}
        return {}

    @staticmethod
    def _document_from_record(record: DocumentRecord) -> _Document:
        return _Document(
            source_id=record.source_id,
            workspace_id=record.workspace_id,
            filename=record.filename,
            file_type=record.file_type,
            content_hash=record.content_hash,
            total_chunks=record.total_chunks,
            ingested_at=record.ingested_at,
            updated_at=record.updated_at,
            metadata_json=record.metadata,
        )

    @staticmethod
    def _record_from_document(row: _Document) -> DocumentRecord:
        return DocumentRecord(
            source_id=row.source_id,
            workspace_id=row.workspace_id,
            filename=row.filename,
            file_type=row.file_type,
            content_hash=row.content_hash,
            total_chunks=row.total_chunks,
            ingested_at=row.ingested_at,
            updated_at=row.updated_at,
            metadata=row.metadata_json or {},
        )

    # ----------------------------------------------------------------- documents
    def upsert_document(self, record: DocumentRecord) -> None:
        with self._session() as session:
            existing = session.get(_Document, record.source_id)
            if existing is None:
                session.add(self._document_from_record(record))
            else:
                for column, value in (
                    ("workspace_id", record.workspace_id),
                    ("filename", record.filename),
                    ("file_type", record.file_type),
                    ("content_hash", record.content_hash),
                    ("total_chunks", record.total_chunks),
                    ("ingested_at", record.ingested_at),
                    ("updated_at", record.updated_at),
                    ("metadata_json", record.metadata),
                ):
                    setattr(existing, column, value)
            session.commit()

    def get_document(self, source_id: str) -> DocumentRecord | None:
        with self._session() as session:
            row = session.get(_Document, source_id)
            return self._record_from_document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._session() as session:
            rows = session.execute(select(_Document).order_by(_Document.source_id)).scalars().all()
            return [self._record_from_document(row) for row in rows]

    def document_count(self) -> int:
        with self._session() as session:
            return int(session.execute(select(func.count()).select_from(_Document)).scalar_one())

    def delete_document(self, source_id: str) -> bool:
        with self._session() as session:
            result = session.execute(select(_Document).where(_Document.source_id == source_id).limit(1)).scalars().all()
            if not result:
                return False
            session.delete(result[0])
            session.commit()
            return True

    def delete_all_documents(self) -> int:
        with self._session() as session:
            rows = session.execute(select(_Document)).scalars().all()
            count = len(rows)
            for row in rows:
                session.delete(row)
            session.commit()
            return count

    # ----------------------------------------------------------------- workspaces
    def upsert_workspace(self, workspace_id: str, name: str) -> None:
        with self._session() as session:
            existing = session.get(_Workspace, workspace_id)
            if existing is None:
                session.add(_Workspace(id=workspace_id, name=name))
            else:
                existing.name = name
            session.commit()

    def get_workspace(self, workspace_id: str) -> str | None:
        with self._session() as session:
            row = session.get(_Workspace, workspace_id)
            return row.name if row else None

    def list_workspaces(self) -> list[tuple[str, str]]:
        with self._session() as session:
            return [(row.id, row.name) for row in session.execute(select(_Workspace)).scalars().all()]

    # ----------------------------------------------------------------- feedback
    def add_feedback(self, source_id: str, question: str, answer: str, rating: int | None = None,
                     comment: str | None = None) -> None:
        with self._session() as session:
            session.add(_Feedback(source_id=source_id, question=question, answer=answer, rating=rating, comment=comment))
            session.commit()

    def list_feedback(self, source_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = select(_Feedback)
            if source_id:
                stmt = stmt.where(_Feedback.source_id == source_id)
            rows = session.execute(stmt.order_by(_Feedback.id.desc()).limit(limit)).scalars().all()
            return [
                {"id": row.id, "source_id": row.source_id, "question": row.question, "answer": row.answer,
                 "rating": row.rating, "comment": row.comment, "created_at": row.created_at}
                for row in rows
            ]

    # ----------------------------------------------------------------- audit
    def add_audit_event(self, action: str, actor: str | None = None, target_id: str | None = None,
                        status: str | None = None, details: dict[str, Any] | None = None) -> None:
        with self._session() as session:
            session.add(_AuditEvent(actor=actor, action=action, target_id=target_id, status=status, details_json=details or {}))
            session.commit()

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.execute(select(_AuditEvent).order_by(_AuditEvent.id.desc()).limit(limit)).scalars().all()
            return [
                {"id": row.id, "actor": row.actor, "action": row.action, "target_id": row.target_id,
                 "status": row.status, "details": row.details_json, "created_at": row.created_at}
                for row in rows
            ]

    # ----------------------------------------------------------------- evals
    def upsert_eval_result(self, suite: str, cases: int, hits: int, mrr: float, passed: bool) -> None:
        with self._session() as session:
            session.add(_EvalResult(suite=suite, cases=cases, hits=hits, mrr=mrr, passed=passed))
            session.commit()

    def list_eval_results(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.execute(select(_EvalResult).order_by(_EvalResult.id.desc()).limit(limit)).scalars().all()
            return [
                {"id": row.id, "suite": row.suite, "cases": row.cases, "hits": row.hits, "mrr": row.mrr,
                 "passed": row.passed, "created_at": row.created_at}
                for row in rows
            ]

    def close(self) -> None:
        self._engine.dispose()


def create_metadata_store(storage_path: str | None = None, db_url: str | None = None) -> SQLAlchemyStore:
    """Create the structured metadata store.

    Default: SQLite beside the chunk store (``rag-metadata.db`` next to the
    storage file); with no storage path, an in-memory SQLite database.
    """
    resolved = db_url
    if resolved is None:
        if storage_path:
            db_file = Path(storage_path).expanduser().parent / "rag-metadata.db"
            resolved = f"sqlite:///{db_file.as_posix()}"
        else:
            resolved = "sqlite://"
    if isinstance(resolved, str) and resolved.startswith("sqlite:///"):
        candidate = resolved[len("sqlite:///"):]
        if candidate and candidate != ":memory:":
            db_file = Path(candidate)
            if not db_file.is_absolute():
                db_file = Path.cwd() / db_file
            db_file.parent.mkdir(parents=True, exist_ok=True)
    return SQLAlchemyStore(resolved)
