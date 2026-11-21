# RAG Knowledge Assistant — Complete Audit Report

**Audit date:** 2026-11-19
**Scope:** Full repository inspection + functional verification (read-only)
**Method:** Every source file read line-by-line; test suite executed in an isolated venv; runtime probes executed against the live app; no application files were modified.

> **Honesty note:** Every claim in this report marked *verified* was reproduced by running code. Claims marked *code-inspected* are derived from reading the implementation. Nothing is credited solely on the basis of README statements.

---

## 1. Repository Layout (actual)

```
RAG-Knowledge-Assistant/                 ← git root (outer wrapper repo)
├── Dockerfile                           ← Render-oriented build (flattens nested project)
├── render.yaml                          ← Render service definition
├── README.md                            ← outer README (contains stale YOUR_USERNAME clone URL)
├── .dockerignore
└── rag-knowledge-assistant-main/        ← the actual application
    ├── src/rag_assistant/
    │   ├── api.py            (523 lines — routes + middleware + metrics + rate limiter + response building)
    │   ├── knowledge_base.py (579 lines — chunking + lexical + semantic + persistence + query)
    │   ├── embeddings.py     (348 lines — 4 embedding providers behind a Protocol)
    │   ├── models.py         (206 lines — Pydantic request/response contracts)
    │   ├── llm.py            (148 lines — urllib-based OpenAI-compatible client + extractive fallback)
    │   ├── config.py         (120 lines — env-driven AppConfig)
    │   ├── document_processor.py (109 lines — PDF/DOCX/TXT/MD parsing)
    │   ├── evals.py          (69 lines — retrieval hit-rate/MRR harness)
    │   ├── cli.py            (37 lines — uvicorn launcher)
    │   └── __init__.py       (version 1.0.0)
    ├── tests/                (5 files, 25 tests)
    ├── ui/index.html         (105 lines — single-file vanilla JS chat UI)
    ├── docs/ARCHITECTURE.md
    ├── Dockerfile, docker-compose.yml, render.yaml, vercel.json, Makefile
    ├── .github/workflows/ci.yml
    ├── pyproject.toml, .env.example, CHANGELOG.md, ROADMAP.md, CONTRIBUTING.md
```

**Duplicate/overlapping files:** two Dockerfiles and two render.yaml files (outer wrapper + inner project) — both functional but a maintenance duplication. No `requirements.txt` (pyproject only — correct). No runtime `data/` directory exists in the checkout (clean state confirmed).

---

## 2. Complete Feature Inventory (verified against code)

| # | Feature | README claim | Actual status | Evidence |
|---|---------|--------------|---------------|----------|
| 1 | TXT/MD ingestion (`/ingest`, `/ingest/file`) | ✅ | **Works** | verified: 201, tests pass |
| 2 | PDF ingestion (pypdf) | ✅ | **Works (text-layer PDFs only)** | code-inspected; no OCR path exists |
| 3 | DOCX ingestion (python-docx, incl. tables→pipe text) | ✅ | **Works** | code-inspected (`document_processor.py:55-71`) |
| 4 | Extension allowlist on upload | ✅ | **Works** | verified: `.exe` → 400 |
| 5 | Chunking: tokens / sentence / paragraph / smart | ✅ | **Works** | tested (`test_smart_chunking…`); strategies validated |
| 6 | TF-IDF lexical retrieval | ✅ | **Works but slow at scale** | verified: 94.6 ms @ 2,000 chunks |
| 7 | Semantic retrieval (4 providers) | ✅ | **Works** (local/local_tfidf tested; ST/ONNX untested in CI) | tests pass w/ numpy |
| 8 | Hybrid + Reciprocal Rank Fusion | ✅ | **Works** | tested, trace includes `rrf_fusion` |
| 9 | Reranker (`term_overlap`, `none`) | ✅ ("lightweight") | **Works — but it is lexical overlap, not a real ranker** | `knowledge_base.py:444-464` |
| 10 | Query trace | ✅ | **Works** | verified in responses |
| 11 | Citations + retrieved context | ✅ | **Works** | tested |
| 12 | Knowledge-gap / evidence gating | ✅ | **Works (heuristic)** | tested; thresholds are hand-tuned (`api.py:284-295`) |
| 13 | Extractive fallback (no LLM key) | ✅ | **Works** | tested |
| 14 | LLM generation (OpenAI-compatible) | ✅ | **Code path exists; NEVER tested, no retry, no streaming** | `llm.py:79-148`; zero tests mock HTTP |
| 15 | Outdated/conflicting-source warnings | ✅ | **Works (heuristic, untested)** | `api.py:308-336` |
| 16 | JSON persistence (atomic write) | ✅ | **Works** | tested (schema v2, tmp+replace) |
| 17 | API-key auth | ✅ | **Works only when key set — OFF by default** | verified: all routes 200 with no key |
| 18 | IP rate limiting | ✅ | **Works** | tested (429) |
| 19 | Request size limits | ✅ | **BROKEN — returns 500 instead of 413/400** | verified: `api.py:216-222` raises inside middleware |
| 20 | Metrics / health / ready | ✅ | **Works** | verified over HTTP |
| 21 | Structured errors + request IDs | ✅ | **Works for route-level errors** | tested |
| 22 | Retrieval eval harness (hit rate, MRR) | ✅ | **Works — retrieval-only** | tested |
| 23 | Metadata filtering | ✅ | **Works — exact-match only** (no operators) | `knowledge_base.py:379-383` |
| 24 | Bulk ingestion | ✅ | **Works but non-atomic** (partial state on failure) | `api.py:455-465` |
| 25 | Web UI (chat, upload, delete, settings, confidence badge) | ✅ | **Works** (code-inspected; XSS escaping is solid) | `ui/index.html` |
| 26 | Docker (non-root, healthcheck) | ✅ | **Works** | Dockerfile reviewed |
| 27 | Conversational memory (`session_id`) | implied | **NOT IMPLEMENTED — field accepted and silently ignored** | verified: response contains no session state |
| 28 | Multi-user / workspaces | roadmap | **NOT IMPLEMENTED** — single global index, single shared key | verified |
| 29 | OCR / scanned PDFs | not claimed | **NOT IMPLEMENTED** — scanned PDF → 400 "No readable text" | `document_processor.py:94-96` |
| 30 | Streaming responses | not claimed | **NOT IMPLEMENTED** | no SSE/StreamingResponse anywhere |
| 31 | Cross-encoder reranking | not claimed | **NOT IMPLEMENTED** | only `term_overlap` |

---

## 3. Verification Results (Phase 4 — actually executed)

### 3.1 Test suite — clean CI environment (reproduces GitHub Actions exactly)
Installed exactly what `ci.yml` installs (`pip install -e '.[dev]'` → no numpy) in an isolated venv:

```
FAILED tests/test_api.py::test_full_query_flow                 — assert 503 == 200
FAILED tests/test_persistence_and_middleware.py::test_api_key_enforcement
FAILED tests/test_persistence_and_middleware.py::test_rate_limit_middleware_blocks_excess_calls
3 failed, 18 passed, 4 skipped
```

**Root cause:** `QueryRequest.embedding_provider` defaults to `"local_tfidf"` (`models.py:67-69`), which requires numpy — and numpy is only in the `[embeddings]` extra, **not** in `[dev]` or base dependencies (`pyproject.toml:14-33`). The UI also defaults to `local_tfidf` (`ui/index.html:53`) and the README example uses it. **On a clean install the documented default query path returns 503.**

### 3.2 Test suite — with numpy added
```
23 passed, 2 skipped
```
The suite itself is sound; it is the *default configuration + dependency set* that is broken.

### 3.3 Lint (CI step 1)
```
ruff 0.17.5 → "Found 125 errors" (exit 1)
```
`ci.yml` runs `ruff check src tests` with unpinned `ruff>=0.6`. The codebase was written against an older rule set; current ruff flags 125 issues (UP006/UP035/I001/RUF007…). **The CI pipeline fails at the lint step before tests even run. CI is red on main.**

### 3.4 Application startup (real uvicorn over HTTP)
```
GET /          → 307 → /ui/
GET /health    → 200 {"status":"ok", ...}
GET /ui/       → 200
GET /docs      → 200
GET /metrics   → 200
```
Startup works. Corrupt persistence file would crash at import (fail-fast — defensible, but no recovery/backup path).

### 3.5 Runtime probes (script executed outside the repo; all storage redirected to temp)
| Probe | Result | Verdict |
|---|---|---|
| Oversized `Content-Length` header | **500** `internal_error` (should be 413) | **BUG** — `api.py:220` raises HTTPException inside HTTP middleware; bypasses the structured exception handlers |
| Non-numeric `Content-Length` | **500** (should be 400) | **BUG** — same cause; full stack trace logged |
| `GET /stats`, `/documents`, `/metrics` with no API key | **200** | auth disabled by default |
| `DELETE /clear` (destructive) with no key | **200** | **SECURITY** — default deploy is wide open |
| Upload `.exe` | 400 | correct |
| `/ingest/file?source_id=<500 chars>` | **201** | query param bypasses the 200-char limit imposed on the JSON route |
| `/ingest/file?chunk_size=-5` | 400 structured | correct (route-level ValueError path works — proving the middleware path is the broken one) |
| `DELETE /documents/does-not-exist` | **200** `removed_chunks:0` | should be 404 |
| `/query` with `session_id` | 200, field ignored | dead API surface |
| `GET /openapi.json` | 200 | docs exposed by default (`RAG_DOCS_ENABLED=true`) |

### 3.6 Performance micro-benchmark (in-memory KB, 2,000 chunks)
| Operation | Measured | Scaling behavior |
|---|---|---|
| Ingest 500 docs → 2,000 chunks | **15.42 s total** | **O(N²)** — every ingest rebuilds doc-freq over the whole corpus (`knowledge_base.py:361-364`) |
| Single ingest into existing 2,000-chunk index | **76.6 ms** | grows linearly with index size |
| TF-IDF query | **94.6 ms** | vectorizes **every chunk on every query** (`knowledge_base.py:392-398`) |
| Hybrid query (local hash) | **199.9 ms** | both legs + RRF + rerank, all pure Python |

**Ceiling estimate:** ~10k chunks → TF-IDF query ≈ 0.5 s+, ingest ≈ 400 ms/doc and climbing. This is the practical scaling wall of the current design (consistent with the honest note in `docs/ARCHITECTURE.md`).

---

## 4. Critical Bugs (confirmed)

1. **CI pipeline is red** — lint (125 errors) fails first, then 3 tests fail on the dependency set CI itself installs. *(Critical — destroys credibility of the entire quality story.)*
2. **Default query configuration fails on clean install** — `local_tfidf` default requires an optional dependency. *(Critical — first-run experience for anyone following the README.)*
3. **Body-limit middleware returns 500 instead of 413/400** — `api.py:216-222` raises `HTTPException` from inside `@app.middleware("http")`, where FastAPI's exception handlers cannot catch it. Structured error contract silently violated; noisy stack traces. *(High)*
4. **Chunked uploads bypass the size guard** — the check reads the `Content-Length` *header* only; a `Transfer-Encoding: chunked` request has none. `/ingest/file` then does `await file.read()` **before** checking length (`api.py:432-434`) → unbounded memory read. *(High, needs auth to reach — but auth is off by default)*
5. **`session_id` accepted and ignored** — misleading API contract; no conversational state exists. *(Medium)*
6. **Dead configuration**: `RAG_RELEVANCE_FLOOR` and `RAG_REQUEST_TIMEOUT_SECONDS` are parsed (`config.py:102-103`) and **never used anywhere**. *(Medium — misleading ops surface)*
7. **`DELETE` unknown source returns 200**; **bulk ingest is non-atomic** (partial index state on mid-batch failure, `api.py:455-465`). *(Medium)*
8. **Version drift**: `pyproject`/`__init__` say `1.0.0`, CHANGELOG stops at `0.2.0`, ROADMAP lists already-implemented features as "mid term". *(Low–Medium)*
9. **`/ingest/file` query params bypass Pydantic validation** — `source_id` unbounded (500 chars accepted), only post-hoc `ValueError` checks. *(Low–Medium)*
10. **Orphaned upload files** — the raw file is written to disk *before* `kb.ingest_with_info()` runs (`api.py:437-444`); if ingestion fails, the artifact remains. *(Low)*

---

## 5. Security Issues (summary — full detail in `SECURITY_AND_PERFORMANCE_REPORT.md`)

| Severity | Issue |
|---|---|
| **Critical** | Authentication **silently disabled** when `RAG_API_KEY` is unset (the default, and the Render template's `sync: false` posture). Verified: destructive `/clear` returns 200 with no key. |
| **Critical** | **No data isolation** — one global index, one shared key. Any key holder sees/deletes every document. Not multi-user ready. |
| **High** | **Indirect prompt injection** — uploaded document text and filenames are placed into the LLM prompt with only a system-prompt-level instruction as defense. No content sanitization, no instruction/data separation, no per-document trust scoring. |
| **High** | API key persisted in `localStorage` (UI) — readable by any XSS. |
| **Medium** | `/metrics`, `/docs`, `/openapi.json` exposed by default in production. |
| **Medium** | No audit trail for destructive operations (`/clear`, deletes). |
| **Medium** | No upload content scanning; extension-based trust; potential PDF/zip-bomb CPU exhaustion (no decompression limits). |
| **Low** | No CSP header (X-Frame-Options DENY + nosniff are set — good). `trust_proxy_headers` takes the *first* XFF hop if enabled (spoofable if misconfigured). |

**Positive security findings (verified):** Pydantic bounds on JSON routes; extension allowlist; non-root Docker user; `.env` gitignored; secrets `sync: false` in render.yaml; rate limiting works; UI escaping (`escapeHtml` applied to all interpolated data) is genuinely solid XSS hygiene.

---

## 6. Code Quality Assessment (Phase 3)

**Strengths**
- Clean module boundaries (`api` / `knowledge_base` / `embeddings` / `llm` / `document_processor` / `models` / `config` / `evals`).
- Embedding providers behind a `Protocol` with lazy imports — genuinely good dependency isolation.
- Thread-safe (`RLock`), atomic persistence (tmp + `replace`), schema versioning constant.
- Typed throughout; Pydantic contracts; structured error envelope with request IDs.
- `docs/ARCHITECTURE.md` is **honest** about the JSON-index scaling boundary — refreshing.
- Tests cover happy paths plus meaningful negatives (401/429/404/storage corruption).

**Weaknesses / debt**
- `api.py` is a god module: middleware, rate limiter, metrics collector, confidence heuristic, source diagnostics, and response assembly all inline. Confidence logic (`api.py:284-295`) is untested hand-tuned arithmetic.
- `KnowledgeBase` monolith: chunking + lexical index + semantic index + persistence + query in one 579-line class; no repository/storage interface (JSON hardcoded).
- **No precomputed lexical index** — TF-IDF vectors recomputed per query; **embeddings not persisted** — full re-encode after every restart or any ingest (any ingest invalidates the *entire* semantic index, `knowledge_base.py:238-239, 363`).
- Global `RLock` held for the whole query including model load and encoding → head-of-line blocking under concurrency.
- Pure-Python cosine loops despite numpy being an available dependency.
- **Test coverage not measured** (no pytest-cov); no type-check, no security scanning in CI.
- Untested: LLM path, bulk, clear, metrics, PDF/DOCX upload, metadata_filter, non-smart chunking, `reranker=none`, middleware limits, concurrency.

---

## 7. Maturity Assessment

| Dimension | Score | Rationale |
|---|---|---|
| Code structure & readability | **7/10** | Well-organized, typed, documented design decisions |
| Test quality (what exists) | **6/10** | Good happy/negative paths; but CI red, coverage unmeasured, LLM path untested |
| RAG sophistication | **5/10** | Hybrid+RRF+rerank+evidence-gate is a solid 2023 baseline; missing modern techniques (cross-encoder, contextual retrieval, query rewriting, citation verification) |
| Security posture | **3/10** | Auth off by default, no isolation, injection surface unmanaged |
| Production readiness | **3/10** | Red CI, JSON store, no streaming, no observability export, no multi-tenancy |
| Overall | **Portfolio-grade MVP, not production** | Honest label: *strong production-style portfolio implementation* (as the README's own scope note says) |

---

## 8. Top 10 Issues to Fix (priority-ranked)

| # | Issue | Priority | Complexity |
|---|---|---|---|
| 1 | CI red: pin/configure ruff + fix lint; add numpy to `[dev]` **or** change default embedder to dependency-free `local` | **Critical** | Low (1 day) |
| 2 | Default query path 503 on clean install (align `QueryRequest` default, UI default, README example) | **Critical** | Low |
| 3 | Auth disabled by default → fail-fast/warn in production mode; protect `/clear` | **Critical** | Low |
| 4 | Middleware body-limit → return `JSONResponse` instead of raising (413/400 contract) + enforce byte-count limit on chunked streams | **High** | Low |
| 5 | No multi-user isolation → workspace model (JWT + metadata-scoped retrieval) | **High** | High |
| 6 | Performance ceiling: persist embeddings / move to vector DB; stop O(N) work per ingest/query | **High** | High |
| 7 | Prompt-injection hardening (delimited untrusted content, sanitization, trust scoring) | **High** | Medium |
| 8 | `session_id` dead field + dead env vars → implement or remove | **Medium** | Low |
| 9 | Bulk ingest atomicity; DELETE 404; upload query-param validation; orphaned-file cleanup | **Medium** | Low |
| 10 | No generation-quality evaluation or citation verification (`grounded=True` is asserted, never proven) | **High** | Medium–High |

---

## 9. Recommended Improvements (condensed — full detail in roadmap)

1. **Fix the foundation** (CI, defaults, middleware, auth posture) — days, not weeks.
2. **Storage upgrade**: Qdrant for vectors (persisted embeddings, ANN, payload filters) + SQLite→PostgreSQL for metadata; migration exporter from the JSON index.
3. **Retrieval upgrade**: real default embeddings (all-MiniLM-L6-v2 — already supported by the codebase), optional cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), query rewriting/multi-query.
4. **Generation upgrade**: streaming (SSE), multi-provider adapters, retries with backoff, implement `session_id` conversational memory.
5. **Trust upgrade**: prompt-injection defenses, citation verification/groundedness scoring, LLM-as-judge eval suite wired into CI, feedback capture.
6. **Production upgrade**: JWT + workspaces, background ingestion with status, structured logs with request IDs, Prometheus metrics, pinned dependencies, coverage/type/security gates.

**UI policy:** the existing `ui/index.html` design is sound (and its XSS escaping is better than average). Preserve it; extend it incrementally (streaming display, feedback buttons, processing status, workspace selector) without redesign.

---

*Companion reports:* `RAG_PIPELINE_ANALYSIS.md` · `SECURITY_AND_PERFORMANCE_REPORT.md` · `PRODUCTION_ARCHITECTURE.md` · `RAG_EVALUATION_STRATEGY.md` · `IMPLEMENTATION_ROADMAP.md`

---

## 11. Phase 0 Remediation Log (executed after approval)

All Phase 0 items were implemented and verified after the report was written. This section records what changed, what was verified, and how.

### 11.1 CI truthfulness (Critical #1)
- **CI replica (exact GitHub Actions environment, no numpy): before → `3 failed, 18 passed, 4 skipped`; after → `21 passed, 4 skipped, ruff: All checks passed`.**
  - The 3 previously failing tests now pass: `test_api.py::test_full_query_flow`, `test_persistence_and_middleware.py::test_api_key_enforcement`, `test_persistence_and_middleware.py::test_rate_limit_middleware_blocks_excess_calls`.
  - The default `/query` path now returns 200 instead of 503 on a clean install (`QueryRequest.embedding_provider` default changed `local_tfidf` → `local`).
- **Lint:** `ruff check src tests` green in both the audit venv (ruff 0.17.5) and the fresh CI replica.
  - Strategy: `ruff check --fix` applied 134 mechanical autofixes (import sorting, `typing.Dict/List/Optional` → modern builtins).
  - 6 residual findings handled explicitly: B008 via `flake8-bugbear.extend-immutable-calls` config for the FastAPI `File/Query/Header/Depends` idiom; RUF007×2 via `itertools.pairwise()`; BLE001×2 (deliberate optional-dependency import guards) via `# noqa`; TRY004 (deliberate `ValueError` contract) via rule exclusion.
  - Pinned: `ruff==0.17.5` in `[dev]`, `[tool.ruff]` with explicit `select = ["E4","E7","E9","F","W","I","UP","B","C4","RUF","PIE"]` so future ruff versions cannot add rules that break CI.
- **Version drift fixed:** `pyproject.toml` 1.0.0 → **1.0.1**, `__version__` → 1.0.1, new `[tool.ruff]` config, new `CHANGELOG.md` `[1.0.1]` entry.

### 11.2 Default-embedder fix (Critical #2)
- `QueryRequest` + `EvalCase`: `embedding_provider` default → `"local"` (`models.py`).
- UI `ui/index.html`: Local Hash is now the first/default `<option>`; cosmetic `statProvider` race removed from `connect()`.
- Docs aligned: inner `README.md` example query now uses `"local"`; new "Enable stronger semantic embeddings" section notes the `[embeddings]` extra and the still-supported `local_tfidf` provider.

### 11.3 Auth posture (Critical #3)
- **Production fail-fast:** `RAG_ENV=production` without `RAG_API_KEY` → `RuntimeError` at import; verified live.
- Warning logged when no key is set.
- `/clear` now requires the `X-Clear-Confirm: DELETE` header → otherwise structured 400. Live-verified: 400 without header, 200 with `X-Clear-Confirm: DELETE`.
- `_guard_api_key` → `hmac.compare_digest`.
- `/documents/{id}` → 404 for unknown sources (live-verified).

### 11.4 Body-limit / size-limit fixes (High #3, #4)
- Oversized `Content-Length` → structured **413** (`payload_too_large`); invalid `Content-Length` → structured **400** (live + probe verified).
- New pure-ASGI `_BodyLimitMiddleware` counts actual streamed bytes (no bypass via chunked transfer), raising `_BodyLimitExceeded`; FastAPI wraps body-read failures in `starlette.HTTPException`, so a new `@app.exception_handler(StarletteHTTPException)` translates the wrapped cause chain to structured 413, and the HTTP middleware translates direct `_BodyLimitExceeded` too.
- `/ingest/file`: query params now validated via FastAPI `Query` bounds (`source_id` ≤ 200, `chunk_size`/`chunk_overlap` within 1–4000/0–4000, overlap < size) — 500-char source_id and `chunk_size=-5` probes now return 422; `_read_upload_limited` reads in 1 MB chunks aborting at the limit; upload artifact is written only after successful ingest (no orphan files).
- Dead config removed: `RAG_REQUEST_TIMEOUT_SECONDS`, `RAG_RELEVANCE_FLOOR` (fields + from_env) deleted; `RAG_ENV` added.

### 11.5 Bulk atomicity & correctness papercuts (Medium #7)
- `KnowledgeBase.ingest_many()`: validate-and-chunk all documents first, then commit under one lock with a single persist → atomic. New `tests/test_phase0_fixes.py::test_bulk_ingest_is_atomic_on_failure` proves the index stays empty when one document is invalid.
- `ingest_with_info` refactored into `_prepare_ingest` + `_build_indexed_chunks` (shared helpers).

### 11.6 Metadata cap + session_id (Medium #5, #8)
- `IngestRequest.metadata` and `QueryRequest.metadata_filter`: new validator capping the serialized JSON size at 64 KB.
- `session_id` documented as "Reserved for future conversational memory; currently accepted but not used."

### 11.7 Docs/config updates
- Inner `README.md`: corrected the `local_tfidf` example → `local`; new "Authentication and production mode" section; API table notes on 404/clear-confirm; project structure reflects reality.
- Outer `README.md`: clone URL fixed to `https://github.com/Ganeshbasani/RAG-Knowledge-Assistant.git`; quick-start paths updated to the real repo layout.
- `.env.example`: dead vars removed, `RAG_ENV` added, production-key comment added.
- Inner `ROADMAP.md`: refreshed to honest remaining items.

### 11.8 Test coverage (Medium #10)
- New `tests/test_phase0_fixes.py` (14 tests): 413/400 contract (declared + streamed), `/clear` confirm, DELETE 404, bulk atomicity (API + KB level), file-param validation, no-orphan-file, default provider, production fail-fast, metadata cap.
- New `tests/conftest.py`: redirects `RAG_STORAGE_PATH`/`RAG_UPLOAD_DIR` to temp so tests stop writing `rag-knowledge-assistant-main/data/` (repo artifacts from earlier runs deleted).
- Existing suites unchanged: `test_persistence_and_middleware.py`, `test_api.py`, etc. all green in both environments.

### 11.9 Verification summary

| Step | Before | After |
|---|---|---|
| CI replica (no numpy): pytest | 3 failed, 18 passed, 4 skipped | **21 passed, 4 skipped** |
| CI replica: ruff 0.17.5 | 125 errors (exit 1) | **All checks passed** |
| Audit venv (numpy): pytest | 23 passed, 2 skipped | **37 passed, 2 skipped** |
| New Phase 0 tests | — | **14 passed** |
| Probe: oversized `Content-Length` | 500 | **413** |
| Probe: invalid `Content-Length` | 500 | **400** |
| Probe: default `/query` | 503 (on clean install) | **200** |
| Probe: `DELETE /clear` (no key) | 200 (destructive!) | **400** |
| Probe: `DELETE /documents/nope` | 200 | **404** |
| Live server: `/`, `/health`, `/ui/`, `/docs`, `/metrics`, `/openapi.json` | 307/200/200/200/200/200 | **all 200** (307→`/ui/`) |
| Live server: `DELETE /clear` without header | — | **400**; with `X-Clear-Confirm: DELETE` → **200** |
| Production import guard | — | `RuntimeError: RAG_ENV=production requires RAG_API_KEY…` |

---

*Companion reports (Phase 1–5 planning):* `PRODUCTION_ARCHITECTURE.md` · `RAG_EVALUATION_STRATEGY.md` · `IMPLEMENTATION_ROADMAP.md`

---

## 12. Phase 1 Remediation Log — Storage & Retrieval Core Upgrade (executed after approval)

All Phase 1 items (1.1–1.8 from `IMPLEMENTATION_ROADMAP.md`) were implemented and verified after the Phase 0 log.

### 12.1 Storage adapter interface (1.1) — `src/rag_assistant/stores.py` (new)
- `VectorStore` protocol: `load_chunks/save_chunks/delete_source/delete_all/count/storage_probe/close/load_vectors/save_vectors/load_vector_key/query_vectors`.
- `JSONVectorStore` — embedded JSON backend (zero deps) preserving the historical on-disk format (`schema_version: 2`, `chunks`) and adding a `vectors` array + `embedding_key` field for Phase 1.5.
- `QdrantVectorStore` — persistent backend (Qdrant, opt-in). Verified against qdrant-client (UUID point ids, `query_points`, payload filters, in-memory `":memory:"` mode for tests).

### 12.2 Qdrant integration (1.2 + 1.3)
- `RAG_VECTOR_STORE=qdrant` (+ `RAG_QDRANT_URL`/`RAG_QDRANT_API_KEY`/`RAG_QDRANT_COLLECTION`).
- **Docker Compose** now ships a `rag-assistant-qdrant` service + `qdrant_data` volume (app defaults to `RAG_VECTOR_STORE=qdrant` in compose).
- `scripts/migrate_json_to_qdrant.py` — reads the JSON index, upserts chunks + vectors, verifies counts + per-chunk parity; `--dry-run` supported.

### 12.3 Embedding persistence + concurrency (1.5)
- Vectors persisted per `(provider, model, dims)` key (`embedding_key`); reload skips re-encode when keys match.
- Provider switch invalidates stored vectors (re-embed, never silent mixing).
- `query_with_trace` now builds/encodes the semantic index **outside** the global lock; retrieval runs under the lock.
- Live server verified: `/health`, `/ingest/async` (202), `/ingest/status/{id}` ("indexed"), `/documents` (SQL metadata), `/ready` — all 200.

### 12.4 Structured metadata store (1.4) — `src/rag_assistant/metadata.py` (new)
- `SQLAlchemyStore` + `DocumentRecord` + `MetadataStore` protocol: documents, workspaces, feedback, audit events, eval results.
- Default: SQLite file (`rag-metadata.db` beside the chunk store; parent dirs auto-created — verified live); `RAG_DB_URL` for PostgreSQL.
- `KnowledgeBase.list_documents()` reads from it; `stats()["documents"]` = SQL count.

### 12.5 Lexical retrieval (1.6) — `src/rag_assistant/ranking.py` (new)
- `BM25Index` — incremental inverted index (add/remove/remove_source/top_k), replaces the full-scan TF-IDF leg of hybrid and the `tfidf` retrieval mode (mode name kept for API compat).
- Ingest is incremental again: no more O(N²) corpus rebuilds.

### 12.6 Cross-encoder reranker (1.7) — `src/rag_assistant/rerankers.py` (new)
- Opt-in: `RAG_RERANKER=cross_encoder` (lazy-loaded `sentence_transformers.CrossEncoder`; needs `[embeddings]`). Default stays `term_overlap`.
- `RerankerMode` Literal in models extended; `tests/test_phase1.py::test_cross_encoder_reranker_scores_pairs` + `test_kb_reranker_cross_encoder_config`.

### 12.7 Background ingestion (1.8) — api.py
- `POST /ingest/async` (202 + `job_id`), `GET /ingest/status/{job_id}` (queued/processing/indexed/failed) backed by FastAPI `BackgroundTasks`.

### 12.8 Bug found & fixed during verification
- SQLite metadata DB used to fail at startup when `data/` didn't exist ("unable to open database file") — fixed by creating parent directories in `create_metadata_store`; live-verified with a fresh data dir.

### 12.9 New tests (all passing)
- `tests/test_stores.py` — JSON backend round-trip/legacy format/delete/corrupt-file + Qdrant in-memory round-trip/vectors/query/delete (4 Qdrant tests skip via `importorskip` when the extra isn't installed).
- `tests/test_ranking.py` — BM25 behavior.
- `tests/test_metadata_store.py` — SQL store CRUD + feedback/audit/eval + KB-backed `list_documents`/`stats`.
- `tests/test_phase1.py` — vector persistence across reload, provider-switch invalidation, BM25 tfidf mode, async ingest flow, migration script (`:memory:` Qdrant).
- Total: **67 passed / 2 skipped** in the audit venv (numpy + extras), **58 passed / 11 skipped, ruff green** in the CI replica (no numpy, no qdrant-client).

### 12.10 Files added/changed (Phase 1)
- New: `src/rag_assistant/stores.py`, `src/rag_assistant/metadata.py`, `src/rag_assistant/ranking.py`, `src/rag_assistant/rerankers.py`, `scripts/migrate_json_to_qdrant.py`, `tests/test_stores.py`, `tests/test_ranking.py`, `tests/test_metadata_store.py`, `tests/test_phase1.py`.
- Changed: `knowledge_base.py` (store-backed), `api.py` (KB wiring + async endpoints), `models.py` (RerankerMode + async models), `config.py` (new env vars), `pyproject.toml` (SQLAlchemy base + `qdrant` extra, version 1.1.0), `docker-compose.yml` (Qdrant service + volume), `README.md`, `ROADMAP.md`, `.env.example`, `CHANGELOG.md` ([1.1.0] entry), `docs/ARCHITECTURE.md` (storage seams).

---

## 13. Phase 2 Remediation Log — Generation & Interaction (executed after approval)

### 13.1 Streaming generation (2.1)
- `POST /query` accepts `"stream": true` → `StreamingResponse` with `text/event-stream`: `delta` events then a `done` event carrying the full `QueryResponse`.
- Gap/no-generate paths send a single `done` event (non-stream responses in `application/json` fall back to the normal `/query`).
- UI (`ui/index.html`): tokens append to the pending message bubble; new "Streaming" capability stat badge (`ReadableStream` feature detection).

### 13.2 LLM provider adapters + retries + token budget (2.2)
- New `llm.py`: `OpenAICompatibleAdapter` (existing default `base_url` flow) + `AnthropicAdapter` (native Messages API, SSE) + `GeminiAdapter` (generateContent, SSE) + `OllamaAdapter` (/api/chat, NDJSON).
- Retries with exponential backoff on 429/5xx (max 3 attempts, `_backoff_sleep` capped at 4s).
- Token-budget enforcement: `RAG_LLM_MAX_OUTPUT_TOKENS` config (default None = unlimited); `AnswerGenerator` facade keeps the legacy public API.

### 13.3 Conversational memory (2.3)
- New `sessions.py`: `SessionStore` (in-memory, TTL-idle expiry on access, last 10 turns, thread-safe).
- `/query` injects history via `kb.augment_with_history()` and records user+assistant turns (`kb.record_turn`).
- New `GET /sessions`, `DELETE /sessions/{id}` endpoints.

### 13.4 Query rewriting (2.4)
- `AnswerGenerator.rewrite_query()`: LLM-based variants via the new adapters; rule-based fallback (`_rule_based_variants`) without an LLM.
- `KnowledgeBase.query_with_trace`: multi-query retrieval per variant, merged with the existing RRF (`queries[:3]` cap).

### 13.5 Context assembly (2.5)
- `_dedup_context` (SequenceMatcher > 0.85), `_strip_page_markers` (`[PAGE n]` stripped from context text; page stays in metadata), `_trim_context` token budget (`QueryRequest.context_max_tokens`, 0 = off).

### 13.6 UI streaming (2.6)
- `ask()` rewritten: streams deltas into the pending bubble when `canStream`; falls back to the non-stream path otherwise.
- New "Streaming" badge in the stats bar.

### 13.7 New tests (audit venv: 86 passed, 2 skipped; CI replica: 77 passed, 11 skipped; ruff green)
- `tests/test_phase2.py` (11): rule-based variants, session store TTL/clear, KB history injection, multi-query trace, context assembly helpers, API streaming events, sessions endpoints.
- `tests/test_llm_adapters.py` (8): adapters with mocked transports — OpenAI parse/retry-on-429/stream-SSE/budget, Anthropic, Gemini, Ollama.

---

## 14. Phase 3 Remediation Log — Trust, Security & Evaluation (executed after approval)

### 14.1 Prompt-injection hardening (3.1)
- `llm._build_messages`: context blocks wrapped in `<<<UNTRUSTED CONTEXT #i ... >>>END UNTRUSTED CONTEXT #i`; question wrapped in `<<<UNTRUSTED QUESTION>>>`; system prompt carries an explicit SECURITY note (untrusted text inside delimiters = data, never instructions).
- Advisory-flags for content (e.g., `content_advisory`) deferred (see notes).

### 14.2 Citation verification (3.2)
- New `api._verify_citations` (lexical): every `[i]` in the answer must exist in citations; cited chunk shares lexical support with the answer (`chunk_token_overlap` ≥ 0.05).
- New `api._grounded_status`: LLM answers get computed `grounded` (all citations exist + supported); extractive answers grounded by construction. Wired into `/query` + streaming (`generation.citation_verification`).

### 14.3 Golden eval datasets (3.3) — `tests/fixtures/eval/`
- `retrieval_cases.jsonl` (10), `answerable_cases.jsonl` (5 with `must_contain`), `unanswerable_cases.jsonl` (6), `adversarial_cases.jsonl` (5), `conflict_cases.jsonl` (2) — hand-verified, versioned.

### 14.4 Extended eval harness (3.4) — `evals.py`
- `load_cases()` (JSONL loader), `recall_at_k()`, `ndcg()`, `abstention_accuracy()` (unanswerable cases with nothing retrieved), `citation_precision()` (expected sources sufficiently represented).

### 14.5 Feedback capture (3.8)
- `models.py`: `FeedbackRequest`, `FeedbackResponse`, `FeedbackListResponse`.
- `KB.add_feedback()`/`KB.list_feedback()` (metadata store); `api.py`: `POST /feedback`, `GET /feedback` (rating 1–5 + comment; rating > 5 → 422).

### 14.6 Audit log for destructive ops + security leftovers (3.9)
- `KB.remove_source()` → `metadata_store.add_audit_event("remove_source", target_id=...)`; `KB.clear()` → `add_audit_event("clear_all")`.
- `api.py`: `Content-Security-Policy` on all responses (`default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'`); docs/OpenAPI disabled when `RAG_API_KEY` is set.

### 14.7 Deferred (Phase 3 notes)
- 3.5 LLM-as-judge (faithfulness/answer-relevance with judge≠generator, 10% human audit) — needs the streaming/retry layer from 2.2; CI gates (coverage ≥ 80%, mypy, bandit, eval regression) — deferred (budget).
- 3.6 Confidence calibration from eval labels (fit logistic/isotonic; gate threshold vs false-answer rate).
- 3.9 upload parse limits (max pages/time) — deferred.
- Advisory flags (`document_processor` content advisories) — deferred.

### 14.8 New tests (audit venv: 95 passed, 2 skipped; CI replica: 86 passed, 11 skipped; ruff green)
- `tests/test_phase3.py` (9): injection markers, citation verify/grounded semantics, eval dataset loading + metric correctness, KB feedback integration, audit events (remove_source/clear_all), CSP header, feedback endpoints.

---

*Phases 0–3 complete. Phase 4 (multi-user workspaces, deploy runbook, load testing, confidence calibration) is pending.*

---

## 15. Phase 4 Remediation Log — Multi-User & Production Readiness (executed after approval)

### 15.1 JWT multi-user auth (4.1)
- New `src/rag_assistant/auth.py`: `create_token()` (sub + workspace_id + exp + iss, HS256 via PyJWT), `verify_token()` (signature + exp + issuer; returns claims or None).
- `config.py`: `RAG_JWT_SECRET`, `RAG_JWT_TTL_SECONDS` (3600), `RAG_JWT_ISSUER`, `RAG_WORKSPACE_DEFAULT` ("default").
- pyproject: `pyjwt>=2.10` added to base deps.

### 15.2 Server-enforced workspace isolation (4.1)
- KB: `query_with_trace(..., workspace_id=)`, `ingest_with_info(..., workspace_id=)` (sets `metadata.workspace_id`), strict `remove_source(..., workspace_id=)`, `list_documents(workspace_id=)`, `workspaces()`; `metadata.py: DocumentRecord.from_chunk_metadata` now writes the `workspace_id` column.
- `api.py`: `_guard_api_key` — static API key (bootstrap admin) OR JWT (claims → `request.state.workspace_id`/`user_id`); routes pass `workspace_id` to KB on `/query`, `/query/semantic`, `/ingest`, `/ingest/file`, `/documents`, `DELETE /documents/{id}`; new `GET /workspaces`, `POST /auth/token` (`TokenRequest` model).
- Security model: client-supplied filters are merged with the server-enforced `workspace_id` filter (`_workspace_filter`), so no one can widen their isolation by forging filters.

### 15.3 Deferred (Phase 4 notes)
- 4.2 per-key/workspace rate limiting (replaces pure-IP keying).
- 4.3 observability (structured logs + Prometheus metrics).
- 4.4 deployment docs (persistent volumes, runbooks).
- 4.5 resilience (graceful shutdown, egress allowlist, backup, load tests).
- 4.6 `/v1` aliases.

### 15.4 New tests
- `tests/test_phase4.py` (4): JWT create/verify round-trip + tamper; auth required (401s); **workspace isolation** (ws-b cannot see/delete ws-a's doc; ws-a still sees it); `/workspaces` + `/auth/token`.

*Phases 0–4 complete.*
