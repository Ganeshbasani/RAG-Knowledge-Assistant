from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw}")


def _env_allowed_origins(raw: str | None) -> Tuple[str, ...]:
    if not raw:
        return ()
    values = tuple(item.strip() for item in raw.split(","))
    return tuple(item for item in values if item)


@dataclass(frozen=True)
class AppConfig:
    service_name: str = "rag-knowledge-assistant"
    api_key: str | None = None
    storage_path: str | None = "./data/rag-index.json"
    upload_dir: str = "./data/uploads"

    default_chunk_size: int = 800
    default_chunk_overlap: int = 120
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    request_body_limit_bytes: int = 10_485_760
    request_timeout_seconds: int = 30
    relevance_floor: float = 0.0

    llm_provider: str = "extractive"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1/chat/completions"
    llm_timeout_seconds: int = 30

    log_level: str = "INFO"
    docs_enabled: bool = True
    allowed_origins: tuple[str, ...] = ()
    trust_proxy_headers: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            service_name=_env("RAG_SERVICE_NAME", "rag-knowledge-assistant"),
            api_key=_env("RAG_API_KEY"),
            storage_path=_normalize_storage_path(_env("RAG_STORAGE_PATH", "./data/rag-index.json")),
            upload_dir=_env("RAG_UPLOAD_DIR", "./data/uploads") or "./data/uploads",
            default_chunk_size=_env_int("RAG_DEFAULT_CHUNK_SIZE", 800),
            default_chunk_overlap=_env_int("RAG_DEFAULT_CHUNK_OVERLAP", 120),
            default_embedding_model=_env("RAG_DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2",
            host=_env("RAG_HOST", "0.0.0.0") or "0.0.0.0",
            port=_env_int("RAG_PORT", 8000),
            reload=_env_bool("RAG_RELOAD", False),
            rate_limit_requests=_env_int("RAG_RATE_LIMIT_REQUESTS", 120),
            rate_limit_window_seconds=_env_int("RAG_RATE_LIMIT_WINDOW_SECONDS", 60),
            request_body_limit_bytes=_env_int("RAG_MAX_REQUEST_BYTES", 10_485_760),
            request_timeout_seconds=_env_int("RAG_REQUEST_TIMEOUT_SECONDS", 30),
            relevance_floor=float(_env("RAG_RELEVANCE_FLOOR", "0.0") or "0.0"),
            llm_provider=_env("RAG_LLM_PROVIDER", "extractive") or "extractive",
            llm_model=_env("RAG_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            llm_api_key=_env("RAG_LLM_API_KEY") or _env("OPENAI_API_KEY"),
            llm_base_url=_env("RAG_LLM_BASE_URL", "https://api.openai.com/v1/chat/completions") or "https://api.openai.com/v1/chat/completions",
            llm_timeout_seconds=_env_int("RAG_LLM_TIMEOUT_SECONDS", 30),
            log_level=_env("RAG_LOG_LEVEL", "INFO") or "INFO",
            docs_enabled=_env_bool("RAG_DOCS_ENABLED", True),
            allowed_origins=_env_allowed_origins(_env("RAG_ALLOWED_ORIGINS")),
            trust_proxy_headers=_env_bool("RAG_TRUST_PROXY_HEADERS", False),
        )


def _normalize_storage_path(raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    return str(path.with_suffix(".json")) if not path.suffix else str(path)
