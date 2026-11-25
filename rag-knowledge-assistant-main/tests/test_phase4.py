"""Phase 4.1: JWT auth + workspace isolation (server-enforced)."""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from rag_assistant.auth import create_token, verify_token
from rag_assistant.config import AppConfig


def test_jwt_create_and_verify(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "test-secret")
    config = AppConfig.from_env()
    token = create_token(user_id="u1", workspace_id="ws-a", app_config=config)
    claims = verify_token(token, app_config=config)
    assert claims is not None
    assert claims["sub"] == "u1"
    assert claims["workspace_id"] == "ws-a"
    assert verify_token("not.a.jwt", app_config=config) is None
    monkeypatch.setenv("RAG_JWT_SECRET", "other-secret")
    assert verify_token(token, app_config=AppConfig.from_env()) is None


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RAG_API_KEY", "admin-key")
    monkeypatch.setenv("RAG_JWT_SECRET", "test-secret")
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    with TestClient(api_module.app) as client:
        yield client, api_module


def test_auth_requires_valid_key_or_token(api_client):
    client, _ = api_client
    assert client.get("/auth/config").json() == {"auth_required": True, "browser_login_enabled": False}
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/stats", headers={"x-api-key": "admin-key"}).status_code == 200


@pytest.fixture
def browser_login_client(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RAG_ENV", "development")
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setenv("RAG_UI_PASSWORD", "demo-passphrase")
    monkeypatch.setenv("RAG_JWT_SECRET", "a-long-enough-browser-session-secret-for-tests")
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    with TestClient(api_module.app) as client:
        yield client, api_module


def test_browser_login_mints_scoped_session_without_exposing_server_api_key(browser_login_client):
    client, _ = browser_login_client
    assert client.get("/auth/config").json() == {"auth_required": True, "browser_login_enabled": True}
    assert client.get("/documents").status_code == 401
    assert client.post("/auth/session", json={"password": "wrong"}).status_code == 401
    login = client.post("/auth/session", json={"password": "demo-passphrase"})
    assert login.status_code == 200
    token = login.json()["token"]
    assert token and "demo-passphrase" not in token
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/documents", headers=headers).status_code == 200
    assert client.post("/auth/token", headers=headers,
                       json={"user_id": "attacker", "workspace_id": "other-workspace"}).status_code == 403


def test_public_demo_reports_auth_disabled_and_accepts_browser_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RAG_ENV", "development")
    for name in ("RAG_API_KEY", "RAG_UI_PASSWORD", "RAG_JWT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")
    with TestClient(api_module.app) as client:
        assert client.get("/auth/config").json() == {"auth_required": False, "browser_login_enabled": False}
        assert client.get("/documents").status_code == 200


def test_jwt_workspace_isolation(api_client):
    client, _ = api_client
    config = AppConfig.from_env()
    token_a = create_token(user_id="u1", workspace_id="ws-a", app_config=config)
    token_b = create_token(user_id="u2", workspace_id="ws-b", app_config=config)
    headers_a = {"x-api-key": token_a}
    headers_b = {"x-api-key": token_b}

    assert client.post("/ingest", json={"source_id": "doc-a", "content": "alpha content"},
                        headers=headers_a).status_code == 201

    docs_a = client.get("/documents", headers=headers_a).json()
    assert [d["source_id"] for d in docs_a["documents"]] == ["doc-a"]

    docs_b = client.get("/documents", headers=headers_b)
    assert docs_b.status_code == 200
    assert docs_b.json()["documents"] == []

    # ws-b cannot delete ws-a's document (server-enforced filter)
    assert client.delete("/documents/doc-a", headers=headers_b).status_code == 404
    # ws-a still sees it
    assert client.get("/documents", headers=headers_a).json()["count"] == 1

    # ws-a can query it
    query = client.post("/query", json={"question": "alpha"}, headers=headers_a)
    assert query.status_code in (200, 404)


def test_workspaces_and_token_endpoints(api_client):
    client, _ = api_client
    admin = {"x-api-key": "admin-key"}
    assert client.get("/workspaces", headers=admin).status_code == 200
    minted = client.post("/auth/token", json={"user_id": "u1", "workspace_id": "ws-1"}, headers=admin)
    assert minted.status_code == 200
    token = minted.json()["token"]
    assert verify_token(token, app_config=AppConfig.from_env())["workspace_id"] == "ws-1"
    # invalid token on an authed route → 401
    assert client.get("/stats", headers={"x-api-key": "not-a-token"}).status_code == 401
