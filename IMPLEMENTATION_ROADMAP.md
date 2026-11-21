# Implementation Roadmap — RAG Knowledge Assistant Upgrade

**Audit date:** 2026-11-19
**Status:** PLANNING ONLY — awaiting approval. No application code has been modified.
**Sequencing principle:** fix truthfulness (CI/defaults) → fix safety (auth/limits) → upgrade storage → upgrade retrieval/generation → upgrade trust → multi-user production → polish.

Effort estimates assume one experienced engineer; complexity: **S** (<1 day) · **M** (1–3 days) · **L** (3–7 days) · **XL** (1–2 weeks).

---

## Phase 0 — Essential Bug Fixes & Truthfulness  *(do first — everything else builds on a green baseline)*

| # | Task | Complexity | Evidence base |
|---|---|---|---|
| 0.1 | **Fix CI lint:** pin `ruff` in `[dev]` (or add `[tool.ruff]` config + `ruff check --fix` pass). Verified: 125 errors with ruff 0.17.5 | S–M | AUDIT §3.3 |
| 0.2 | **Fix default query path:** change `QueryRequest.embedding_provider` default to dependency-free `local`; align UI default select + README example; **or** add `numpy` to `[dev]` (chosen: default→`local`, keeps clean installs lean) | S | AUDIT §3.1 |
| 0.3 | **Middleware error contract:** return `JSONResponse` for 413/400 instead of raising inside middleware | S | verified 500s |
| 0.4 | **Streamed body-size enforcement** (byte counter during read; fixes chunked-transfer bypass) | M | SECURITY HIGH-4 |
| 0.5 | **Auth posture:** production mode refuses to start (or hard-warns) without `RAG_API_KEY`; `/clear` requires confirmation token; constant-time key compare | S–M | SECURITY CRITICAL-1 |
| 0.6 | **File-route validation:** Pydantic model for query params (`source_id ≤200`, chunk bounds); write upload only after successful ingest (or cleanup on failure) | S | verified 500-char source_id |
| 0.7 | **Correctness papercuts:** `DELETE` unknown → 404; bulk ingest validate-then-commit (atomic); remove/repurpose dead config (`RAG_RELEVANCE_FLOOR`, `RAG_REQUEST_TIMEOUT_SECONDS`); remove or implement `session_id` | S–M | AUDIT §4 |
| 0.8 | **Version hygiene:** CHANGELOG entry for 1.0.0; ROADMAP refresh (remove already-built items); fix outer README `YOUR_USERNAME` clone URL | S | AUDIT §4.9 |
| 0.9 | **Dependency lockfile** (uv/poetry) + pin CI actions/tools — prevents recurrence of the ruff drift | M | verified CI drift |
| 0.10 | **Test additions catching every bug above** (body-limit, upload params, 404, bulk atomicity) + pytest-cov baseline | M | AUDIT §6 |

**Exit criteria:** CI green on clean checkout; README quickstart works verbatim; no verified 500-on-abuse-control; auth cannot be silently off in production.
**Total: ~1–1.5 weeks · Complexity: Low overall**

---

## Phase 1 — Storage & Retrieval Core Upgrade  *(the biggest engineering value)*

| # | Task | Complexity |
|---|---|---|
| 1.1 | **Storage adapter interface** extracted from `KnowledgeBase` (keeps all public methods as the compatibility seam) | M |
| 1.2 | **Qdrant integration:** chunks+vectors persisted; payload `workspace_id`/metadata filters; Docker Compose service + volume | L |
| 1.3 | **JSON→Qdrant migration script** with verification (counts, query parity smoke test); JSON engine retained as embedded fallback for tests/dev | M |
| 1.4 | **Metadata store:** SQLAlchemy models (documents, versions, users/workspaces, feedback, audit, eval_results) — SQLite now, Postgres-ready | L |
| 1.5 | **Embedding defaults:** production default `all-MiniLM-L6-v2`; persist vectors keyed by `(model, dims)`; explicit re-embed on model change; load/encode **outside** the global lock | M |
| 1.6 | **Kill O(N²):** incremental doc-freq updates; replace full-scan TF-IDF with BM25 index (`rank_bm25` or own inverted index) | M |
| 1.7 | **Cross-encoder reranker** (opt-in `cross-encoder/ms-marco-MiniLM-L-6-v2`), rerank top-20→top-5, keep `term_overlap` default | M |
| 1.8 | **Background ingestion** (BackgroundTasks) + `GET /ingest/status/{id}` + UI status chip | M |

**Exit criteria:** p95 retrieval <100 ms @10k chunks (from measured ~200 ms @2k with worse scaling); restart does not re-embed; single ingest O(chunk); Compose brings up app+Qdrant with one command.
**Total: ~2–2.5 weeks · Complexity: Medium-High**

---

## Phase 2 — Generation & Interaction Upgrades

| # | Task | Complexity |
|---|---|---|
| 2.1 | **Streaming answers (SSE)** through FastAPI `StreamingResponse`; UI renders tokens incrementally (same message bubble) | M |
| 2.2 | **LLM provider adapters** (OpenAI-compatible refactor + Anthropic/Gemini/Ollama), retries with exponential backoff on 429/5xx, structured error logging (no secrets), circuit breaker | L |
| 2.3 | **Implement conversational memory** (make `session_id` real): session store with TTL, last-N-turns context injection, "clear conversation" — **or remove the field** (decision: implement; it's expected of modern assistants) | M |
| 2.4 | **Query rewriting / multi-query retrieval:** LLM (or rule-based fallback) expands the question → parallel retrieval → RRF (existing fusion reused) | M |
| 2.5 | **Context assembly:** token budget enforcement, near-duplicate chunk collapse, strip `[PAGE]` markers from scored text (keep in metadata) | M |
| 2.6 | **Targeted UI additions:** streaming display, expandable trace/latency row, capability detection for embedder/LLM options | S–M |

**Exit criteria:** first token <1.5 s perceived latency; follow-up questions resolve pronouns; LLM failures degrade to extractive with visible warning (existing behavior preserved).
**Total: ~1.5–2 weeks · Complexity: Medium**

---

## Phase 3 — Trust, Security & Evaluation  *(the interview-differentiating layer)*

| # | Task | Complexity |
|---|---|---|
| 3.1 | **Prompt-injection hardening:** delimited untrusted blocks, instruction-priority rules, content advisory flags, per-document trust level | M |
| 3.2 | **Citation verification:** every `[i]` must exist + sentence↔chunk entailment (lexical first, judge second) → `grounded` becomes computed | M–L |
| 3.3 | **Golden eval datasets** (retrieval/answerable/unanswerable/adversarial/conflict suites) — see `RAG_EVALUATION_STRATEGY.md` | M |
| 3.4 | **Harness extension:** recall@k, nDCG, abstention accuracy, citation precision, latency percentiles; keep hit-rate/MRR API shape | M |
| 3.5 | **LLM-as-judge** faithfulness/answer-relevance (temperature 0, JSON output, judge≠generator, 10% human audit) | L |
| 3.6 | **Confidence calibration** from eval labels; gate threshold set to target false-answer rate | M |
| 3.7 | **CI gates:** coverage ≥80%, mypy, bandit, eval regression >2pts fails; nightly judge job | M |
| 3.8 | **Feedback capture:** thumbs up/down + comment → `feedback` table → joins eval dataset | S–M |
| 3.9 | **Security hardening leftovers:** CSP header, docs off when authed, audit log for `/clear`+deletes, upload parse limits (pages/time) | M |

**Exit criteria:** all trust claims (grounded, knowledge-gap, citations) backed by measured metrics in CI; injection suite ≥90%; security report findings Critical/High all closed.
**Total: ~2–2.5 weeks · Complexity: Medium-High**

---

## Phase 4 — Multi-User & Production Readiness

| # | Task | Complexity |
|---|---|---|
| 4.1 | **AuthN/Z:** JWT (PyJWT) + OAuth2 password flow; workspace model; static key becomes bootstrap admin; server-enforced `workspace_id` metadata filter on every query/ingest/delete | L |
| 4.2 | **Rate limiting per key/workspace** (replace pure-IP keying; XFF policy per deploy target) | M |
| 4.3 | **Observability:** structured JSON logs with `request_id` correlation; Prometheus metrics (latency histograms, stage timings, LLM fallback counters, queue depth); optional OTel tracing | L |
| 4.4 | **Deployment:** persistent volumes documented (Qdrant/SQLite/uploads — Render ephemeral-disk data-loss warning), health/readiness wired to new deps, `RAG_ENV=production` profile, Postgres migration runbook | M |
| 4.5 | **Resilience:** graceful shutdown drains, LLM egress allowlist option, index backup/restore script, load test (Locust) vs. SLOs | M |
| 4.6 | **API versioning:** `/v1` aliases; OpenAPI examples; deprecation policy documented | S |

**Exit criteria:** two isolated workspaces proven by test (user A cannot retrieve user B's docs even with crafted filters); p95 and error-rate SLOs verified under load; deploy runbook reproduces from clean machine.
**Total: ~2 weeks · Complexity: High**

---

## Phase 5 — Portfolio & Documentation Polish

| # | Task | Complexity |
|---|---|---|
| 5.1 | README refresh: real screenshots/GIF of current UI, accurate quickstart (verified), benchmark numbers from eval runs, honest scope note (keep the existing honesty) | M |
| 5.2 | Architecture diagram update (current → target), ADRs for Qdrant/Postgres/streaming decisions | S |
| 5.3 | CHANGELOG/ROADMAP/version alignment; CONTRIBUTING updated for new gates | S |
| 5.4 | Demo dataset + seeded eval so reviewers run `docker compose up` and immediately see cited answers + eval report | M |
| 5.5 | "Interview narrative" doc: 5-minute story of design decisions (hybrid RRF, evidence gate, calibration, tenancy) with measured numbers | S |

**Total: ~3–5 days · Complexity: Low**

---

## Priority Summary

| Priority | Items | Why |
|---|---|---|
| **Critical** (start immediately) | Phase 0 entirely; 1.2–1.5 (vector store + defaults + persistence) | Broken CI destroys credibility; JSON store is the scaling wall; auth-off is a live risk |
| **High** | Phase 1.6–1.8; Phase 2.1–2.3; Phase 3.1–3.5 | Quality, UX, and the trust story |
| **Medium** | Phase 2.4–2.6; 3.6–3.9; Phase 4 | Differentiation and production claims |
| **Low** | Phase 5; optional extras (OCR, PDF tables, API embeddings) | Polish; do last or defer |

---

## Dependency Graph

```
Phase 0 (CI/defaults/auth) ──► Phase 1 (storage/retrieval) ──► Phase 2 (generation)
        │                              │                            │
        │                              ▼                            ▼
        └─────────────────────► Phase 3 (trust/evals) ◄─────────────┘
                                        │
                                        ▼
                                 Phase 4 (multi-user prod) ──► Phase 5 (polish)
```
- Phase 3 needs Phase 1 (stable retrieval to measure) and Phase 2 (real LLM path to judge).
- Phase 4 depends on Phase 1.4 (metadata models) for workspaces.
- Phase 5 runs last but README fixes for *false statements* happen in Phase 0 (truthfulness first).

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qdrant adds ops burden for a portfolio project | Medium | Keep JSON fallback; Compose profile; single-node default |
| sentence-transformers/torch image size (~2 GB) | Medium | ONNX path already exists — use it for slim images; optional extra |
| Scope creep (OCR, agents, LangChain…) | High | Non-goals list in `PRODUCTION_ARCHITECTURE.md` is binding |
| Eval judge costs in CI | Medium | Free-tier Layers 1/3/4 in PR CI; judge runs nightly |
| Breaking the UI while upgrading API | Low | Compatibility seam: keep routes/models; UI changes additive only |
| Multi-user refactor destabilizes retrieval | Medium | Workspace filter added as default-empty metadata filter; parity tests |

---

## Suggested First Milestone (approval gate)

**"Phase 0 + Qdrant spike"** — one week: CI green, README verified verbatim, middleware/auth fixed, plus a time-boxed Qdrant adapter spike proving query parity against the JSON engine. At the review gate, decide: proceed to full Phase 1 or re-scope.

*Awaiting your approval before any implementation begins.*
