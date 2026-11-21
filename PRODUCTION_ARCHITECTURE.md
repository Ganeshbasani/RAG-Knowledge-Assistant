# Production Architecture — Target Design for the RAG Knowledge Assistant

**Audit date:** 2026-11-19
**Status:** PROPOSED — planning only. No application code has been modified.
**Principle:** evolve the existing architecture; preserve the UI and API contracts; add interfaces where the current monolith blocks scale, without a rewrite.

---

## 1. Current vs. Target — What Changes and Why

```
CURRENT (verified)                              TARGET (proposed)
─────────────────────                           ─────────────────────────
Browser UI (index.html)          same ───────►  Browser UI (same design, extended in place)
      │                                                │
FastAPI (api.py: routes +                    API Gateway layer (FastAPI)
  middleware + metrics +                     ├─ AuthN/Z: JWT + API keys (workspace claims)
  confidence heuristics inline)              ├─ Middleware: body limits (streamed), rate limits
      │                                      ├─ /v1 versioned routes (existing paths kept as aliases)
KnowledgeBase (monolith)                     │
  ├─ chunking                                Orchestrator / service layer
  ├─ TF-IDF scan (O(N) query)                ├─ IngestionService (background jobs + status)
  ├─ in-memory embeddings (not persisted)    ├─ QueryService (retrieve → fuse → rerank → gate)
  ├─ JSON persistence (O(N) write)           ├─ GenerationService (providers, streaming, retries)
  └─ global RLock                            └─ EvaluationService (offline + online metrics)
      │                                             │
LLM via urllib (no retry/stream)              Retrieval pipeline (pure functions, no I/O)
                                              ├─ BM25/lexical index  (rank_bm25 or own)
extractive fallback                           ├─ Vector index        (Qdrant — persisted ANN)
                                              ├─ RRF fusion (existing logic, reused)
                                              ├─ Reranker: term_overlap (default) | cross-encoder (opt)
                                              └─ metadata filter (workspace-scoped, server-enforced)
JSON file index                               │
                                              Storage
                                              ├─ Qdrant          → chunks + vectors + payload filters
                                              ├─ SQLite→Postgres → documents, versions, users,
                                              │                    workspaces, feedback, audit log
                                              └─ Object store/dir→ raw uploaded files (after successful
                                                                     ingest; hash-addressed, GC-able)
                                              LLM layer
                                              ├─ Provider adapters: OpenAI-compatible (existing client),
                                              │   Anthropic, Gemini, local Ollama
                                              └─ retries (429/5xx backoff), timeouts, streaming SSE
```

**Design rule:** `KnowledgeBase`'s public methods (`ingest_with_info`, `query_with_trace`, `remove_source`, `list_documents`, `stats`, `health`) are the **compatibility seam**. New storage engines implement the same interface behind it, so routes, tests, and the UI remain valid during migration.

---

## 2. Component Design

### 2.1 API layer
- Keep all current routes and response models (`QueryResponse` etc.) — the UI depends on them. Add `/v1/…` aliases; deprecate nothing in this phase.
- Move confidence/diagnostic helpers (`_confidence`, `_diagnose_sources`, `_source_citations`) out of `api.py` into a `grounding.py` service — testable in isolation.
- Middleware fixes: return `JSONResponse` (never raise) for 413/400; enforce streamed byte limits.
- Auth: keep the static `RAG_API_KEY` as a **bootstrap admin key**; add JWT (PyJWT) with `workspace_id` claims for clients; every retrieval call injects `workspace_id` as a **mandatory** metadata filter (never client-controlled).

### 2.2 Ingestion pipeline (async, observable)
```
POST /ingest/file ─► validate (Pydantic query model) ─► parse (existing document_processor)
        ─► dedupe by SHA-256 ─► [background task] chunk (existing strategies)
        ─► embed (batched, provider per workspace policy) ─► upsert Qdrant + metadata row
        ─► status: queued → processing → indexed | failed(reason)
```
- Start with FastAPI `BackgroundTasks` (zero new infrastructure); swap to ARQ/Celery + Redis only when queue depth/retry requirements demand it.
- Keep synchronous `/ingest` for small payloads and backward compatibility.

### 2.3 Query pipeline (unchanged semantics, faster stages)
```
question (+ optional rewritten variants) ─► metadata-filtered retrieval
   ├─ BM25 leg (replaces hand-rolled TF-IDF scan)
   ├─ Qdrant ANN leg (persisted embeddings)
   ▼
RRF fusion (existing implementation reused verbatim)
   ▼
reranker: term_overlap (default) | cross-encoder/ms-marco-MiniLM-L-6-v2 (opt-in)
   ▼
evidence gate (existing logic, thresholds calibrated via evals — see RAG_EVALUATION_STRATEGY.md)
   ▼
context assembly (token-budgeted, duplicate-collapsed) ─► GenerationService
   ├─ streaming SSE answer + inline citations
   ├─ citation verification (every [i] must exist; sentence↔chunk entailment check)
   ▼
QueryResponse { answer, grounded*, confidence, citations, warnings, trace, generation }
```
\* `grounded` becomes **computed** (verified citations + gate passed), not assumed.

### 2.4 Storage decisions & migration
| Store | Role | Why this choice |
|---|---|---|
| **Qdrant** | Vectors + chunk payload | Purpose-built ANN (HNSW), payload indexes enable metadata filtering *during* retrieval (not after), single-container self-host, first-class Docker Compose, filtering API maps 1:1 to the existing `metadata_filter`. Managed mode available if ops burden must be avoided. |
| **SQLite → PostgreSQL** | Documents, versions, users/workspaces, feedback, audit log, eval results | SQLite for the single-node portfolio deployment (zero ops, file-backed, backup = copy file); same SQLAlchemy models migrate to Postgres for multi-user production. Avoids running two network services in dev. |
| **JSON engine (keep)** | Test/dev fallback | Existing tests run without infrastructure; embedded fallback preserves the "clone and run" README promise. |
| *Rejected:* pgvector as primary | — | Viable if Postgres-only is a hard constraint, but weaker ANN performance/ops features than Qdrant at scale; adds extension management. |
| *Rejected:* Chroma | — | Weaker production posture (auth, filtering, clustering) for the same operational cost. |
| *Rejected:* Milvus/Weaviate | — | Heavier operational footprint than this project needs. |

**Migration path:** `scripts/migrate_json_to_qdrant.py` — read existing `rag-index.json`, embed+upsert, verify counts, dual-read smoke test, then flip config. No data loss; JSON file remains as rollback.

### 2.5 Embedding policy
- **Production default:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-friendly, already implemented in `embeddings.py`) — chosen for latency/quality balance on typical hardware.
- **Dev/CI default:** `local` (hash) — dependency-free so clean installs work (fixes verified CI failure). `local_tfidf` remains available but never the default.
- Vectors stored **with a `(model, dims)` key**; changing models triggers explicit re-embed (versioned), never silent mixing.
- Optional later: API embeddings (OpenAI `text-embedding-3-small`, Voyage) behind the same `EmbeddingProvider` Protocol — the abstraction already exists.

### 2.6 LLM layer
- Thin provider adapters (OpenAI-compatible — refactor existing `llm.py` client; Anthropic; Gemini; Ollama) behind a `Generator` Protocol with: timeout, exponential backoff on 429/5xx, structured error codes logged (never keys), token-budget enforcement, and **streaming** support.
- No heavyweight orchestrator framework (LangChain/LlamaIndex) — the pipeline is small, custom code stays debuggable and interview-explainable; add a dependency only when it replaces substantial maintained code.

### 2.7 Observability
- Structured JSON logs with `request_id` propagated from middleware (currently set but not logged).
- Prometheus-format `/metrics` (request latency histograms, retrieval stage timings, LLM latency/fallback counts, queue depth) — the existing `_MetricsCollector` becomes the source for counters.
- Optional OpenTelemetry traces spanning retrieve→rerank→generate (the existing `trace` dict is the natural span payload).

### 2.8 Deployment topology
```
                 ┌────────────────────────────────────────────┐
                 │  Container host (Render / Docker Compose)  │
   Client ─────► │  app (uvicorn, non-root) :8000             │
   (UI+API)      │    ├─ /ui  static (existing index.html)    │
                 │    └─ /v1  API                             │
                 │  qdrant :6333 (volume-backed)              │
                 │  app volume: /app/data (uploads + sqlite)  │
                 └────────────────────────────────────────────┘
```
- Single-node Compose profile for portfolio; document that horizontal scale requires external Qdrant/Postgres (the honest scaling note from `docs/ARCHITECTURE.md` carries over).
- Persist volumes for Qdrant + uploads + SQLite; Render ephemeral disk must be explicitly noted as data-loss-risk (current latent issue).
- Keep Dockerfile hardening (non-root, healthcheck); add `.env`-required check for production mode.

---

## 3. UI Preservation Plan (explicit constraint)

**The existing `ui/index.html` is not to be redesigned.** It is compact, responsive, and its XSS escaping is solid. Targeted, additive improvements only:

| Change | Why | Risk to existing look |
|---|---|---|
| Streamed answer rendering (append tokens to the pending message) | First-token latency UX | None — same message bubble |
| Thumbs up/down + comment under answers | Feedback → eval dataset | Additive controls in existing styles |
| Upload status chip (queued→processing→indexed) | Async ingestion visibility | Reuses existing status/doc components |
| Workspace selector in toolbar (when multi-user lands) | Isolation | New control in existing toolbar grid |
| Capability detection for embedder/LLM options | Hide "Sentence Transformers" when extra not installed | Same `<select>` |
| Show trace/latency in an expandable "details" row | Debuggability for demos | Collapsed by default |
| Minor: `statProvider` overwritten by "Ready" on connect (cosmetic race, `ui/index.html:90`) | Buglet | One-line fix |

Everything else (layout, CSS variables, light theme, card design, chat panel) stays.

---

## 4. Recommended Technology Choices (with rationale)

| Concern | Choice | Why appropriate | Alternatives considered |
|---|---|---|---|
| Vector DB | **Qdrant** | ANN at scale, payload filtering aligned with existing `metadata_filter`, easy self-host, active maintenance | pgvector (OK if Postgres-only), Chroma (weaker prod features), Milvus (too heavy here) |
| Metadata DB | **SQLite (dev) → PostgreSQL (prod)** via SQLAlchemy | Zero-ops start, direct upgrade path, models shared | Raw JSON (current, limited), Mongo (no relational need) |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** (opt-in) | ~70 MB, CPU-capable, standard benchmark model | bge-reranker (larger), LLM-judge rerank (cost/latency) |
| Default embedder | **all-MiniLM-L6-v2** | Already supported in codebase; fast CPU inference; industry baseline | bge-small (slightly better quality, heavier), API embeddings (cost/latency) |
| LLM access | **Thin custom adapters** over OpenAI-compatible API (+ Anthropic/Gemini) | Existing urllib client already proves the pattern; keeps deps minimal | LiteLLM (adds dep for modest gain), LangChain (too much abstraction) |
| Background jobs | **FastAPI BackgroundTasks → ARQ+Redis** only if needed | No new infra at first; clear upgrade trigger (queue depth/retries) | Celery (heavy), Dramatiq |
| Streaming | **SSE via StreamingResponse** | Native FastAPI; UI uses `fetch` + `ReadableStream` | WebSockets (overkill, no bidirectional need) |
| Auth | **PyJWT + OAuth2 password flow**; keep static admin key | Standard, explainable, workspace claims map to metadata filters | OAuth2 proxies/Keycloak (overkill for scope) |
| Eval | **Own harness (extend `evals.py`) + LLM-as-judge** using existing LLM client | No heavy framework; CI-runnable; honest about metrics | RAGAS (optional later — heavy deps), DeepEval |
| Dependencies | **uv lockfile + pinned CI**, ruff/mypy/bandit/pytest-cov gates | Directly addresses the verified CI drift failure | poetry (fine too; uv is faster/simpler) |

---

## 5. Non-Goals (this upgrade)

- ❌ Rewriting the UI or changing its visual design.
- ❌ Replacing FastAPI or restructuring routes the UI already calls.
- ❌ Adopting LangChain/LlamaIndex as the orchestration core.
- ❌ Kubernetes/microservice decomposition — a modular monolith is the correct size for this system.
- ❌ Real-time collaborative editing, recommendations, or agent tool-use — out of scope until the core is production-grade.

---

## 6. Target Architecture Scorecard

| Property | Now | After upgrade |
|---|---|---|
| Query latency @10k chunks | ~0.5–1 s | <100 ms (ANN + BM25) |
| Embedding persistence | ❌ re-embed every restart | ✅ persisted, versioned |
| Multi-user isolation | ❌ | ✅ JWT workspaces + enforced filters |
| Streaming | ❌ | ✅ SSE |
| Groundedness | asserted | verified (citation checks + evals) |
| CI | ❌ red | ✅ lint+types+security+tests+eval gate |
| Horizontal scale | single process only | app replicas + external Qdrant/Postgres |
| UI | preserved | preserved + additive improvements |
