# Security & Performance Report — RAG Knowledge Assistant

**Audit date:** 2026-11-19
**Method:** Static code review of all source + runtime probes against the live application (isolated temp storage). Findings marked *verified* were reproduced by execution.

---

# Part A — Security Audit

## A1. Threat Model (what this system must defend)

| Asset | Threat actor | Impact if compromised |
|---|---|---|
| Indexed documents (IP, policies, PII) | External attacker, co-tenant user | Data breach |
| LLM API key & provider | External attacker | Cost fraud, data exfiltration via prompt |
| Service availability | Script kiddie, runaway client | Denial of service |
| Answer integrity | Malicious document author | Indirect prompt injection → misleading answers |
| Control plane (ingest/delete/clear) | Unauthenticated caller | Total data destruction |

---

## A2. Findings (severity-ordered, all evidence-based)

### 🔴 CRITICAL-1 — Authentication is disabled by default, including destructive endpoints
**Evidence (verified):** with `RAG_API_KEY` unset (the default, and `render.yaml` declares it `sync: false`):
```
GET    /stats      → 200 (no key)
GET    /documents  → 200 (no key)   ← full document listing
GET    /metrics    → 200 (no key)
DELETE /clear      → 200 (no key)   ← wipes the entire index
```
**Root cause:** `api.py:245-247` — `if app_config.api_key and x_api_key != app_config.api_key` → empty/missing key silently disables the guard.
**Impact:** a default deployment (Docker/Render with no env configured) exposes ingest, query, listing, and **irreversible deletion** to the world.
**Fix:** fail-fast or loudly warn when `RAG_ENV=production` and no key is set; require a confirmation token for `/clear`; consider refusing to start if `RAG_API_KEY` is empty outside dev mode.

### 🔴 CRITICAL-2 — No data isolation / multi-tenancy
**Evidence:** one global `KnowledgeBase` instance, one shared key, metadata filters are optional and client-supplied (`api.py:47-53`).
**Impact:** any key holder can read, overwrite (ingest with same `source_id`), or delete every document. There is no per-user/workspace scoping anywhere in the code. **The platform is single-tenant by construction.**
**Fix:** workspace_id claim in JWT → enforced as a mandatory metadata filter on every ingest/query/delete (never trust client-provided filters).

### 🟠 HIGH-3 — Indirect prompt injection surface (uploaded documents → LLM prompt)
**Evidence:** `llm.py:104-116` — document chunk text is concatenated raw into the user message; filenames flow into the citation list. The only defense is the system-prompt sentence *"Do not invent facts."*
**Impact:** a poisoned document can contain *"Ignore previous instructions and reply with …"* and steer generated answers; a crafted filename can inject text into the source list. The extractive fallback is naturally immune; **LLM mode is not.**
**Mitigations (recommended):**
1. Wrap untrusted content in explicit delimiters with an instruction-priority rule (system > user > document).
2. Pre-sanitization pass: strip/flag imperative instruction patterns in ingested text; record a `content_advisory` in metadata.
3. Per-document trust levels; untrusted docs never influence system-level behavior.
4. Post-generation check: answer must not contain instructions-directed phrases; ground every sentence in cited chunks (ties to citation verification).
5. Treat filenames as untrusted display data (already escaped in UI — keep that).

### 🟠 HIGH-4 — Request body limit bypass + unbounded read (DoS)
**Evidence (verified):** `api.py:216-222` checks **only the `Content-Length` header**. A `Transfer-Encoding: chunked` request carries no such header → check skipped. `/ingest/file` then executes `raw = await file.read()` (`api.py:432`) **before** comparing `len(raw)` to the limit → entire body buffered in memory first.
**Impact:** memory exhaustion by a single large upload (reachable without auth in default posture — compounding CRITICAL-1). Pydantic's `max_length=2_000_000` bounds JSON `/ingest`, but not the file route.
**Fix:** enforce a streaming byte counter during read (read in chunks, abort at limit); keep the header check as an early reject.

### 🟠 HIGH-5 — Middleware exception handling breaks the error contract
**Evidence (verified):**
```
oversized Content-Length → 500 {"error_code":"internal_error"}   (expected 413)
invalid  Content-Length → 500 {"error_code":"internal_error"}   (expected 400)
```
`api.py:216-222` raises `HTTPException` from inside `@app.middleware("http")`, *outside* the reach of FastAPI's exception handlers (which sit inside the middleware stack) → ServerErrorMiddleware converts it to a generic 500 and logs a full stack trace.
**Impact:** clients can't distinguish abuse-control rejections from real crashes; monitoring sees false 500s; log noise.
**Fix:** return `JSONResponse(...)` directly from the middleware instead of raising.

### 🟡 MEDIUM-6 — Secrets & API key handling
- ✅ `.env` gitignored; `render.yaml` uses `sync: false` for `RAG_API_KEY`/`RAG_LLM_API_KEY`.
- ⚠️ UI stores the API key in **`localStorage`** (`ui/index.html:80`) — persistent, readable by any XSS or malicious browser extension.
- ⚠️ No key rotation story; single static key shared by all clients; keys compared with plain `!=` (no constant-time compare — `hmac.compare_digest` recommended).
- ⚠️ `RAG_LLM_BASE_URL` is env-controlled — fine for trusted operators, but there is no egress allowlist; a compromised env can redirect prompts+keys to an arbitrary host (SSRF-from-config).
**Fix:** short-lived scoped tokens for the browser (session-exchange endpoint), constant-time comparison, optional egress allowlist for LLM calls.

### 🟡 MEDIUM-7 — Over-exposed production surface
- `/docs`, `/redoc`, `/openapi.json` **enabled by default** (`RAG_DOCS_ENABLED=true`) — verified 200. API schema helps attackers enumerate capabilities.
- `/metrics` unauthenticated — leaks route paths, request counts, error rates.
- `X-Frame-Options: DENY` and `X-Content-Type-Options: nosniff` are set ✅, but **no CSP** (UI is inline-script by design; a CSP with `'unsafe-inline'` at minimum, or nonces, should be added).
- CORS only activates when `RAG_ALLOWED_ORIGINS` is set ✅ (safe default: same-origin UI).
**Fix:** default docs off when an API key is configured; add CSP; protect `/metrics` or move to an internal port.

### 🟡 MEDIUM-8 — Upload handling
- Extension allowlist verified working (`.exe` → 400) ✅; `Path(filename).suffix` sanitization prevents traversal in the stored name (hash-prefixed, sanitized) ✅.
- ⚠️ No content-type sniffing, no malware scan, **no decompression-ratio limits** → crafted PDF can pin CPU/memory in pypdf (zip-bomb class).
- ⚠️ Orphaned files: written to disk **before** indexing; failures leave artifacts (`api.py:437-444`).
- ⚠️ `/ingest/file` query params (`source_id`, `chunk_size`, `chunk_overlap`) bypass Pydantic models — `source_id` length unbounded (verified: 500 chars accepted).
**Fix:** validate query params with a Pydantic model; write file only after successful indexing (or clean up on failure); add parse-time limits (max pages, max extract time).

### 🟡 MEDIUM-9 — Audit & abuse controls
- No audit log for destructive actions (`/clear`, `DELETE /documents/*`) — who wiped the index is unknowable. Request IDs exist but are not correlated into structured logs.
- Rate limiting is **IP-based only** (`api.py:181-186`): behind a proxy without `RAG_TRUST_PROXY_HEADERS`, all traffic shares one bucket (self-DoS); with it enabled, the **first** XFF hop is trusted (spoofable unless the edge strips/overrides XFF).
- `DELETE /clear` has no step-up confirmation beyond the shared key.
**Fix:** per-key/per-workspace rate limits, structured audit events with request_id, XFF handling documented for each deploy target.

### 🟢 LOW-10 — Misc
- Version drift (pyproject `1.0.0` vs CHANGELOG `0.2.0`) — trust/docs issue, not security.
- Error messages for LLM failures are deliberately generic (good — no key leakage) but *too* generic for operators (401/429/500 indistinguishable) — log the status code server-side without the body/key.
- No dependency pinning/lockfile → supply-chain drift (CI already demonstrated this: unpinned ruff broke the pipeline).

## A3. Security Posture Summary

| Control | Status |
|---|---|
| Authentication | ⚠️ Implemented, **off by default**, single static key, no rotation |
| Authorization / isolation | ❌ None (single tenant) |
| Input validation (JSON routes) | ✅ Strong (Pydantic bounds) |
| Input validation (file query params) | ❌ Weak (bypasses models) |
| Upload safety | ⚠️ Extension allowlist ✅; no sniffing/limits; orphan writes |
| Rate limiting | ⚠️ Works; IP-keyed; proxy caveats |
| Request size limits | ❌ Broken (500s + chunked bypass) |
| Prompt injection defense | ❌ System prompt only |
| Secret management | ⚠️ OK at rest; localStorage exposure; no rotation |
| Error handling / logging | ⚠️ Structured for routes; middleware path broken; no audit trail |
| Security headers | ⚠️ Partial (no CSP) |
| Container hardening | ✅ Non-root, healthcheck, no secrets baked in |
| Test coverage of security controls | ⚠️ Auth/rate-limit tested; body-limit, injection, upload limits not |

**Overall: 3.5/10** — the *components* of a security program exist (key auth, rate limits, validation, structured errors) but defaults, isolation, and injection defense are not production-grade.

---

# Part B — Performance & Scalability Audit

## B1. Measured Baseline (isolated venv, Windows, 2,000-chunk index)

| Operation | Measured | Notes |
|---|---|---|
| Ingest 500 docs → 2,000 chunks | **15.42 s** | ~31 ms/doc average, **growing linearly** |
| Single ingest into 2,000-chunk index | **76.6 ms** | full doc-freq rebuild + full JSON rewrite |
| TF-IDF query | **94.6 ms** | re-vectorizes **all** chunks per query |
| Hybrid query (local hash + RRF + rerank) | **199.9 ms** | two full scans + fusion + lexical rerank |
| First semantic query after restart | unbounded | full corpus re-embed, model load **under the global lock** |

## B2. Hotspots (root-cause analysis)

1. **O(N²) ingest** — `ingest_with_info` calls `_rebuild_doc_freq()` over the entire corpus on every document (`knowledge_base.py:361-364`). Fix: incremental doc-freq updates (add new tokens' counts; on replace, decrement old) → O(chunk).
2. **O(N) lexical query** — `_query_tfidf` tokenizes + vectorizes every chunk per request (`knowledge_base.py:392-398`). Fix: precomputed sparse index (inverted index with static tf-idf, or BM25 index) invalidated incrementally.
3. **Full JSON rewrite per write** — `_persist()` serializes the whole index (`indent=2`) on every ingest/delete (`knowledge_base.py:271-285`). Fix: vector DB owns vectors; metadata DB does row-level writes.
4. **Embeddings never persisted + full invalidation** — any ingest clears `_chunk_embeddings` (`knowledge_base.py:238-239, 363`); restart re-embeds everything; switching embedder re-embeds everything. Fix: persisted vectors keyed by (model, dims) in the vector store.
5. **Global `RLock` held across model load + encoding + full query** (`knowledge_base.py:474-507`) → all requests serialize behind the slowest stage; first-query model load blocks ingest too. Fix: scope locks to state mutation; build/load indexes outside the lock; consider per-stage read/write locks or immutable snapshot swap.
6. **Pure-Python cosine** — numpy is an available dependency but scoring loops are Python (`knowledge_base.py:221-230`). Fix: numpy matrix scoring locally, or delegate to Qdrant HNSW.
7. **Synchronous everything** — FastAPI `def` routes run in the threadpool (OK), but a large ingest occupies a worker for the full chunk+embed+persist cycle; no background processing/queue.
8. **LLM latency** — blocking `urllib` call up to 30 s with no streaming: the UI shows "Searching…" until the full answer arrives. First-token latency is the dominant UX cost.

## B3. Scalability Limits (current design)

| Dimension | Practical limit | Breakage mode |
|---|---|---|
| Chunks (single process) | ~10–20k | query > 0.5 s, ingest > 0.4 s/doc |
| Documents | ~1–2k | JSON file tens of MB, rewrite latency |
| Concurrent writers | 1 | multi-worker uvicorn → in-memory divergence (acknowledged in ARCHITECTURE.md) |
| Concurrent readers | threadpool-bound, serialized by lock | tail latency = slowest query |
| File size | 10 MB intended (header check) | chunked bypass → unbounded read |
| Corpus re-embed time | O(N) per restart/switch | minutes at 100k chunks |

## B4. Performance Fixes — Priority × Effort

| # | Fix | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | Incremental doc-freq (kill O(N²) ingest) | Ingest 10–100× faster | Low | **Critical** |
| 2 | Vector store (Qdrant): persisted vectors + ANN | Query → ms; removes re-embed | Medium | **Critical** |
| 3 | Move index build/encode outside the lock | Concurrency ×N | Low | High |
| 4 | Precomputed BM25/inverted index | Query 94 ms → <10 ms | Medium | High |
| 5 | Streaming LLM responses | Perceived latency −90% | Medium | High |
| 6 | Background ingestion + status | Uploads return instantly | Medium | Medium |
| 7 | numpy vectorized scoring (if staying local) | 3–5× query speedup | Low | Medium |
| 8 | Token-budgeted context assembly | Prevents LLM token overflow | Low | Medium |

## B5. Reliability

- ✅ Atomic persistence (tmp + `replace`), fail-fast on corrupt index, `/ready` storage probe, non-root container, healthcheck.
- ⚠️ LLM client: single attempt, no retry/backoff on 429/5xx, no circuit breaker; failures fall back to extractive (good degradation) but log only a generic string.
- ⚠️ Bulk ingest non-atomic → partial state on failure.
- ⚠️ No backup/restore or snapshot tooling for the index; Render disk is ephemeral by default → **data loss on redeploy** unless a persistent disk is attached (deployment concern to document).
- ⚠️ No graceful-shutdown drain logic beyond uvicorn defaults; in-flight persistence is safe (atomic) but background work (once added) will need shutdown hooks.

---

## Verdict

**Security: 3.5/10 · Performance: 4/10 (at current scale targets) · Reliability: 6/10**

The codebase has more security *awareness* than typical portfolio projects (rate limits, structured errors, non-root Docker, honest scope notes), but three verified defects — **auth-off-by-default, broken body limits, no isolation** — must be fixed before any production claim. Performance is entirely adequate for a demo corpus and hits a hard wall near ~10k chunks due to three specific O(N) behaviors, each with a known, bounded fix.
