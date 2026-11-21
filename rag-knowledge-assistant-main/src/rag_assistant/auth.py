"""JWT multi-user auth (Phase 4.1): workspace claims + server-enforced isolation.

Tokens are minted with :func:`create_token` and carry ``sub`` (user id) +
``workspace_id``. Every route that reads or writes the index receives the
caller's workspace via ``request.state.workspace_id`` and the KB applies a
server-side metadata filter (:func:`rag_assistant.knowledge_base.KnowledgeBase._workspace_filter`),
so clients cannot widen their isolation by forging filters.
"""

from __future__ import annotations

import time
from typing import Any

import jwt

from rag_assistant.config import AppConfig

_JWT_ALGORITHM = "HS256"


def _app_config() -> AppConfig:
    return AppConfig.from_env()


def create_token(*, user_id: str, workspace_id: str, app_config: AppConfig | None = None) -> str:
    """Mint a JWT for a user + workspace (used by the token-minting endpoint)."""
    config = app_config or _app_config()
    if not config.jwt_secret:
        raise RuntimeError("RAG_JWT_SECRET is not configured")
    now = int(time.time())
    return jwt.encode(
        {
            "sub": user_id,
            "workspace_id": workspace_id,
            "iat": now,
            "exp": now + config.jwt_token_ttl_seconds,
            "iss": config.jwt_issuer,
        },
        config.jwt_secret,
        algorithm=_JWT_ALGORITHM,
    )


def verify_token(token: str | None, app_config: AppConfig | None = None) -> dict[str, Any] | None:
    """Verify a JWT and return its claims, or None when invalid/expired."""
    config = app_config or _app_config()
    if not token or not config.jwt_secret:
        return None
    try:
        return jwt.decode(
            token,
            config.jwt_secret,
            algorithms=[_JWT_ALGORITHM],
            options={"verify_signature": True, "verify_exp": True, "verify_iss": True, "require": ["exp", "sub"]},
            issuer=config.jwt_issuer,
        )
    except jwt.PyJWTError:
        return None
