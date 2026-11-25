from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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


def _env_allowed_origins(raw: str | None) -> tuple[str, ...]:
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
    environment: str = "development"

    # Storage (Phase 1)
    vector_store: str = "json"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag-chunks"
    db_url: str | None = None

    # Reranking (Phase 1.7)
    reranker: str = "term_overlap"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation (Phase 2.2)
    llm_max_output_tokens: int | None = None

    # Multi-user auth (Phase 4.1)
    jwt_secret: str | None = None
    ui_password: str | None = None
    jwt_token_ttl_seconds: int = 3600
    jwt_issuer: str = "rag-knowledge-assistant"
    workspace_default: str = "default"

    llm_provider: str = "openrouter"
    llm_model: str = "openrouter/free"
    llm_api_key: str | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_timeout_seconds: int = 30
    llm_connect_timeout_seconds: int = 8
    llm_read_timeout_seconds: int = 24
    llm_max_retries: int = 2
    llm_query_rewrite_enabled: bool = False
    llm_fallback_models: tuple[str, ...] = ()

    log_level: str = "INFO"
    docs_enabled: bool = True
    allowed_origins: tuple[str, ...] = ()
    trust_proxy_headers: bool = False

    @classmethod
    def from_env(cls) -> AppConfig:
        llm_provider = (_env("RAG_LLM_PROVIDER", "openrouter") or "openrouter").strip().lower()
        if llm_provider == "gemini":
            # Keep Google credentials separate so an old OpenAI key is never
            # accidentally sent to Google's endpoint.
            llm_api_key = _env("GEMINI_API_KEY")
        elif llm_provider == "openrouter":
            # Accept the provider-native name and the application's generic secret name.
            llm_api_key = _env("OPENROUTER_API_KEY") or _env("RAG_LLM_API_KEY")
        else:
            llm_api_key = _env("RAG_LLM_API_KEY") or _env("OPENAI_API_KEY")
        default_llm_model = "openrouter/free" if llm_provider == "openrouter" else "gemini-3.7-flash"
        default_llm_base_url = (
            "https://openrouter.ai/api/v1/chat/completions"
            if llm_provider == "openrouter" else "https://api.openai.com/v1/chat/completions"
        )
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
            environment=(_env("RAG_ENV", "development") or "development").strip().lower(),
            vector_store=(_env("RAG_VECTOR_STORE", "json") or "json").strip().lower(),
            qdrant_url=_env("RAG_QDRANT_URL") or _env("QDRANT_URL"),
            qdrant_api_key=_env("RAG_QDRANT_API_KEY") or _env("QDRANT_API_KEY"),
            qdrant_collection=_env("RAG_QDRANT_COLLECTION", "rag-chunks") or "rag-chunks",
            db_url=_env("RAG_DB_URL"),
            reranker=(_env("RAG_RERANKER", "term_overlap") or "term_overlap").strip().lower(),
            reranker_model=_env("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            llm_max_output_tokens=(_env_int("RAG_LLM_MAX_OUTPUT_TOKENS", 0) or None),
            jwt_secret=_env("RAG_JWT_SECRET"),
            ui_password=_env("RAG_UI_PASSWORD"),
            jwt_token_ttl_seconds=_env_int("RAG_JWT_TTL_SECONDS", 3600),
            jwt_issuer=_env("RAG_JWT_ISSUER", "rag-knowledge-assistant") or "rag-knowledge-assistant",
            workspace_default=_env("RAG_WORKSPACE_DEFAULT", "default") or "default",
            llm_provider=llm_provider,
            llm_model=_env("RAG_LLM_MODEL", default_llm_model) or default_llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=_env("RAG_LLM_BASE_URL", default_llm_base_url) or default_llm_base_url,
            llm_timeout_seconds=_env_int("RAG_LLM_TIMEOUT_SECONDS", 30),
            llm_connect_timeout_seconds=_env_int("RAG_LLM_CONNECT_TIMEOUT_SECONDS", 8),
            llm_read_timeout_seconds=_env_int("RAG_LLM_READ_TIMEOUT_SECONDS", 24),
            llm_max_retries=_env_int("RAG_LLM_MAX_RETRIES", 2),
            llm_query_rewrite_enabled=_env_bool("RAG_LLM_QUERY_REWRITE_ENABLED", False),
            llm_fallback_models=tuple(
                model.strip() for model in (_env("RAG_LLM_FALLBACK_MODELS", "") or "").split(",") if model.strip()
            ),
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
