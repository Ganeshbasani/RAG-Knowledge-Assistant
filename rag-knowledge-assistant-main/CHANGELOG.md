# Changelog

## [1.4.1] - 2026-11-21

- **Per-key/workspace rate limiting (4.2):** rate-limit identity is now the caller's workspace claim (JWT), user id, or hashed API key (bootstrap admin is shared) — falling back to client IP (trust_proxy_headers) for anonymous callers.


## [1.4.0] - 2026-11-21

- **JWT multi-user auth (4.1):** `RAG_JWT_SECRET`; `create_token`/`verify_token` (HS256, exp, issuer); `POST /auth/token` (mint); `_guard_api_key` accepts bootstrap admin API key OR JWT with `workspace_id` claim.
- **Server-enforced workspace isolation (4.1):** KB workspace filters — `query_with_trace(..., workspace_id=)`, `ingest_with_info(..., workspace_id=)`, strict `remove_source(..., workspace_id=)` (404 for others), `list_documents(workspace_id=)`; `DocumentRecord.workspace_id`; `GET /workspaces`.
- **Dependency:** `pyjwt>=2.10` added to base deps.
- Deferred (budget): 4.2 per-key/workspace rate limiting, 4.3 observability, 4.4 deployment runbooks, 4.5 resilience/backup/load tests, 4.6 /v1 aliases.

# Changelog

## [1.4.0] - 2026-11-21

- **JWT multi-user auth (4.1):** `RAG_JWT_SECRET`; `create_token`/`verify_token` (HS256, exp, issuer); `POST /auth/token` (mint); `_guard_api_key` accepts bootstrap admin API key OR JWT with `workspace_id` claim.
- **Server-enforced workspace isolation (4.1):** KB workspace filters � `query_with_trace(..., workspace_id=)`, `ingest_with_info(..., workspace_id=)`, strict `remove_source(..., workspace_id=)` (404 for others), `list_documents(workspace_id=)`; metadata records store `workspace_id`; `GET /workspaces`; `GET /documents` returns only the caller's workspace (legacy docs without workspace visible when no JWT).
- **Dependency:** `pyjwt>=2.10` added to base deps.
- Deferred (budget): per-key/workspace rate limiting (4.2), observability (4.3), deployment docs/runbooks (4.4), resilience/backup/load tests (4.5).


## [1.3.0] - 2026-11-20

- **Prompt-injection hardening (3.1):** all untrusted content (question + document excerpts) is wrapped in `<<<UNTRUSTED ... >>>` delimiters, and the system prompt carries an explicit instruction-priority/security rule (treat in-delimiter text as data, never instructions).
- **Citation verification (3.2):** `grounded` is now computed for LLM answers — every citation must exist and the cited chunk must share lexical support with the answer (NLI/judge verification deferred); audit trail for destructive ops.
- **Golden eval datasets (3.3):** `tests/fixtures/eval/` — retrieval, answerable, unanswerable, adversarial, conflict cases (hand-verified).
- **Harness extension (3.4):** `evals.load_cases()`, `recall@k`, `ndcg`, `abstention_accuracy`, `citation_precision`.
- **Feedback capture (3.8):** `POST /feedback` / `GET /feedback` (rating, comment); feedback rows persist via the metadata store.
- **Security leftovers (3.9):** audit log rows for `remove_source`/`clear_all`; CSP header on all responses; docs/OpenAPI disabled when an API key is configured.
- Deferred (see Phase 3 notes): LLM-as-judge, CI gates (mypy/bandit/coverage/eval-regression), advisory flags.

## [1.2.0] - 2026-11-20

- **Streaming generation (2.1):** `POST /query` accepts `"stream": true` → SSE (`text/event-stream`) with `delta` events followed by a `done` event carrying the full `QueryResponse`.
- **LLM provider adapters (2.2):** provider refactor — `openai_compatible` (existing `base_url` flow) + new `anthropic` (native Messages API, SSE), `gemini` (`generateContent`, SSE), `ollama` (`/api/chat`, NDJSON). Exponential-backoff retries on 429/5xx (max 3 attempts), token-budget enforcement (`RAG_LLM_MAX_OUTPUT_TOKENS`), structured errors without leaking secrets.
- **Conversational memory (2.3):** `SessionStore` (in-memory, TTL-bounded, last 10 turns) — `/query` injects history (`session_id` now functional), records user+assistant turns; new `GET /sessions`, `DELETE /sessions/{id}`.
- **Query rewriting (2.4):** `AnswerGenerator.rewrite_query` — LLM-based variants when enabled, rule-based fallback otherwise; KB `query_with_trace` runs multi-query retrieval and merges via the existing RRF.
- **Context assembly (2.5):** dedup of near-duplicate context chunks (SequenceMatcher > 0.85), `[PAGE n]` markers stripped from context text (page stays in metadata), optional `context_max_tokens` budget on `QueryRequest`.
- **UI (2.1, 2.6):** incremental streaming display (deltas appended to the pending message bubble), "Streaming" capability stat, history display via sessions endpoints.
- New tests: `tests/test_phase2.py`, `tests/test_llm_adapters.py` (unit tests of provider adapters with mocked transports).

## [1.1.0] - 2026-11-20

- **Storage engine (Phase 1):** new `VectorStore` seam extracted from `KnowledgeBase`; JSON backend (zero-dep, default) + **Qdrant** backend (`RAG_VECTOR_STORE=qdrant`) with persisted chunks + vectors, UUID point ids, payload filters, Docker Compose service + volume; `scripts/migrate_json_to_qdrant.py` migration script with parity verification.
- **Structured metadata store (1.4):** `SQLAlchemyStore` — documents, feedback, audit events, eval results — SQLite by default (`rag-metadata.db` beside the index), PostgreSQL via `RAG_DB_URL`; `KnowledgeBase.list_documents()` reads from it.
- **Embedding persistence (1.5):** vectors are persisted (keyed by provider/model/dims) and loaded on restart; provider switches invalidate and re-embed; encode runs outside the global lock.
- **Lexical retrieval (1.6):** incremental BM25 index replaces the full-scan TF-IDF scan; ingest is O(chunk) again (no O(N²) corpus rebuilds); new tests in `tests/test_ranking.py`, `tests/test_stores.py`.
- **Cross-encoder reranker (1.7):** opt-in via `RAG_RERANKER=cross_encoder` (needs `[embeddings]`); `tests/test_phase1.py::test_cross_encoder_reranker_scores_pairs`.
- **Background ingestion (1.8):** `POST /ingest/async` (202 + job id) + `GET /ingest/status/{job_id}`; FastAPI `BackgroundTasks` (ARQ/Redis upgrade only when queue depth requires).
- Version bump 1.0.1 → 1.1.0.

## [1.0.1] - 2026-11-20

- Fixed CI: pinned `ruff` to 0.17.5 and added a deliberate `[tool.ruff]` rule set; `ruff check src tests` is green again.
- Fixed the default query path on clean installs: `QueryRequest`/`EvalCase` now default `embedding_provider` to the dependency-free `local` provider (was `local_tfidf`, which 503s without numpy); UI and docs aligned.
- Middleware/auth fixes: oversized and invalid `Content-Length` now return structured 413/400 (previously 500s); added a streamed body-limit middleware so chunked uploads cannot bypass the limit.
- Production fail-fast: `RAG_ENV=production` without `RAG_API_KEY` refuses to start; warning logged when authentication is off.
- `/clear` now requires the `X-Clear-Confirm: DELETE` header; `DELETE /documents/{id}` returns 404 for unknown sources.
- `/ingest/file` query params validated (`source_id` ≤ 200 chars, chunk bounds); upload artifacts written only after successful ingest (no orphans).
- Bulk ingest is atomic (validate-then-commit); dead config (`RAG_REQUEST_TIMEOUT_SECONDS`, `RAG_RELEVANCE_FLOOR`) removed; `RAG_ENV` added to `AppConfig`.
- Metadata on ingested documents is capped at 64KB serialized; `session_id` field is documented as reserved/unused.
- New tests: `tests/test_phase0_fixes.py`, `tests/conftest.py` (tests no longer write repo `data/`).

## [0.2.0] - 2026-03-04

- Added persistence support with file-backed storage for indexed chunks.
- Added config/env loading (`AppConfig`) for deployment settings.
- Added optional static API key auth and rate limiting middleware.
- Added input abuse controls via request size checks.
- Added structured error responses and global error handlers.
- Added observability via request metrics, request ID headers, and `/metrics`.
- Added improved health endpoints: `/health` and `/ready`.
- Hardened local embedding providers (`local_tfidf`, `onnx_local`).

## [0.1.0]

- Initial minimal RAG retrieval implementation.
- In-memory storage and TF-IDF/semantic query baseline.
- FastAPI endpoints for ingest, semantic query, and chunk stats.
