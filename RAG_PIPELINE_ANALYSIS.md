# RAG Pipeline Analysis — Stage-by-Stage Evaluation

**Audit date:** 2026-11-19
**Scope:** End-to-end analysis of the actual retrieval-augmented generation pipeline in `rag-knowledge-assistant-main/`.
Each stage documents: *current implementation → verified behavior → limitations → recommended upgrade (feasibility × value)*.

---

## Pipeline Map (as implemented)

```
Upload/Text ─► document_processor ─► KnowledgeBase.chunking ─► JSON persistence
                                                                    │
Question ─► QueryRequest ─► [TF-IDF leg] ─┐                          │
                                          ├─ RRF fusion ─► term_overlap rerank ─► evidence gate
              [Semantic leg] ─────────────┘                          │
                                                    ┌───────────────┴───────────────┐
                                                    ▼                               ▼
                                          confidence < 0.35 / no evidence    citations + context
                                                    │                               │
                                                    ▼                               ▼
                                          knowledge-gap response          LLM generate (or extractive)
                                                                                │
                                                                                ▼
                                                              QueryResponse {answer, grounded, citations, trace}
```

---

## Stage 1 — Document Ingestion & Parsing

**Implementation:** `document_processor.py` — extension allowlist (`.pdf .txt .md .markdown .docx`), SHA-256 content hash, sanitized default `source_id`, page markers `[PAGE n]` for PDFs, DOCX tables flattened to ` | ` rows.

**Verified behavior:**
- `.exe` upload correctly rejected (400) — *verified*.
- Unsupported/empty/unreadable documents produce structured 400s.
- `/ingest/file` accepts an **unvalidated `source_id` query parameter** (500 chars accepted — *verified*), bypassing the 200-char Pydantic limit used by `/ingest`.
- Raw file is written to `upload_dir` **before** indexing succeeds → orphaned artifacts on failure.

**Limitations:**
- **No OCR** — scanned/image PDFs fail with "No readable text" (correct error, missing capability).
- **No image/table extraction from PDFs** (DOCX tables only).
- No HTML/CSV/PPTX; no language detection; no de-duplication across *different filenames with identical content* (hash is embedded in the default source_id, so same-name+same-content is idempotent, but same-content/different-name creates duplicates).
- **No document version history** — re-ingest with `replace_existing=True` discards the previous version entirely.
- Extension-based trust only; no content-type sniffing or malware scanning.

**Upgrade options:**

| Option | Feasibility | Value | Verdict |
|---|---|---|---|
| OCR via PyMuPDF+Tesseract / OCRmyPDF (optional extra) | Medium (heavy deps) | High for real docs | **Do — behind `[ocr]` extra** |
| PDF table extraction (pdfplumber/camelot) | Medium | Medium | Phase 2 |
| Content-hash dedupe across filenames | High | Medium | **Do — cheap** |
| Version history (keep prior versions in metadata store) | High | Medium | **Do with metadata DB** |
| Background processing with status polling | High | High | **Do (see Stage 4/roadmap)** |

---

## Stage 2 — Chunking

**Implementation:** `knowledge_base.py:133-196` — four strategies: `tokens` (sliding window), `sentence`, `paragraph`, `smart` (paragraphs ≤ chunk_size/2 kept whole, else split to sentences, then pack with token-budget + unit-level overlap).

**Verified:** smart chunking test passes (`indexed >= 2`, sentence boundaries preserved, strategy recorded in metadata). All strategies validated against overlap < chunk_size.

**Strengths:** genuinely better than naive fixed-size splitting; boundaries respected; overlap implemented at unit level; per-request `chunk_size/overlap/strategy` supported with validation.

**Limitations:**
- No structure awareness beyond paragraphs (no heading/section detection, no markdown-aware splitting despite `.md` support).
- No semantic chunking (embedding-similarity boundaries) and no small-to-big retrieval (parent-document chunks) — retrievals return the raw chunk, which limits context quality.
- `[PAGE n]` markers become *tokens* in the text — they pollute TF-IDF/term-overlap scoring slightly and surface in UI context.
- Chunk metadata lacks section titles/neighbors, which weakens reranking and citation labels.

**Upgrade options:** markdown/heading-aware splitter (high value, low effort) → parent-child chunk retrieval (high value, medium effort) → semantic chunking (medium value, only worth it with persisted embeddings).

---

## Stage 3 — Embedding Generation

**Implementation:** `embeddings.py` — four providers behind one `Protocol`: `LocalHashProvider` (char-hash, zero deps), `LocalTfIdfProvider` (numpy), `SentenceTransformerProvider` (lazy import), `OnnxTransformerProvider` (onnxruntime+transformers).

**Verified:**
- Clean install (`[dev]` only) → default `local_tfidf` provider raises → **default `/query` returns 503** (reproduced by 3 failing CI tests).
- Provider cache key includes provider+model+dims; switching providers **invalidates and re-encodes the entire corpus**.
- **Embeddings are never persisted** — `_chunk_embeddings` lives only in memory (`knowledge_base.py:61`); every restart re-embeds all chunks on first semantic query.
- Model loading and full-corpus encoding happen **while holding the global query lock** → first query after ingest/restart blocks all other requests.

**Strengths:** clean abstraction; deterministic local providers keep tests dependency-free; ONNX path shows engineering depth.

**Limitations:** default provider choice is broken on clean installs; no embedding persistence; no batched async encoding; no dimension/re-embedding migration story (changing models silently produces incompatible vectors — mitigated only by the cache key, not by persisted-vector invalidation since none are persisted).

**Upgrade options:**

| Option | Feasibility | Value | Verdict |
|---|---|---|---|
| Make `all-MiniLM-L6-v2` the production default (already supported!) and `local` (hash) the dependency-free test default | High | **Critical** | **Do first** |
| Persist vectors in Qdrant (they become the vector store) | High | **Critical** | **Do** |
| Batch/async encoding + load models outside the lock | High | High | **Do** |
| API embeddings (OpenAI `text-embedding-3-small`, Voyage) behind the same Protocol | High | Medium | Optional extra |

---

## Stage 4 — Vector Storage & Persistence

**Implementation:** single JSON file (`schema_version: 2`), atomic tmp+replace writes, doc-freq rebuilt on load, corruption → `KnowledgeBaseStorageError` at startup (fail-fast).

**Verified:** save/reload round-trip test passes; probe file used by `/ready`; full-file rewrite on **every** ingest.

**Limitations (the core scaling constraint):**
- **O(N) rewrite per ingest → O(N²) cumulative**; measured **15.4 s to ingest 500 docs / 2,000 chunks**; a single post-index ingest costs 76.6 ms at 2k chunks and grows linearly.
- No ANN index — every query is a linear scan.
- No concurrent-writer safety beyond the process lock (two workers = two in-memory copies clobbering each other). Multi-worker uvicorn would break — correctly acknowledged in `docs/ARCHITECTURE.md`.
- No backup/restore, no export/import tooling, no pagination for large listings.

**Upgrade options:** **Qdrant** (recommended: ANN, payload filtering, persistent, docker-compose ready, free self-host) — high feasibility since `KnowledgeBase` is the single integration point. Alternative: **pgvector** if a PostgreSQL metadata store is adopted anyway (fewer moving parts, weaker ANN at large scale). Chroma/Milvilab are weaker production postures; keep the JSON engine as an embedded fallback for tests/dev.

---

## Stage 5 — Retrieval (lexical, semantic, hybrid)

**Implementation:** `knowledge_base.py:385-442`.
- **TF-IDF:** custom `tf * idf` vectors + cosine — but **re-vectorizes every chunk on every query** (measured **94.6 ms @ 2,000 chunks**).
- **Semantic:** cosine over the in-memory embedding matrix — **pure-Python loops** (measured hybrid total **199.9 ms**).
- **Hybrid:** Reciprocal Rank Fusion (k=60) over both ranked lists — textbook-correct implementation, trace exposes each stage.

**Strengths:** RRF is implemented correctly; deterministic; metadata filter applied pre-scoring; candidate pool size configurable; trace makes behavior debuggable — genuinely good for interviews.

**Limitations:** no precomputed sparse index (no inverted index / BM25 — BM25 would also *improve* quality over raw tf-idf); min_score semantics differ per leg (RRF scores vs cosine scores are on different scales — the fused score ignores `min_score` entirely, applying it only pre-fusion per leg); no ANN; no query/document expansion; no filtering *post*-fusion with score thresholds.

**Upgrade options:** BM25 (rank_bm25 or own inverted index — high value, low effort) → persisted vector ANN via Qdrant (critical) → **query rewriting / multi-query retrieval** (LLM rewrites + parallel retrieval + fusion — high value, medium effort) → contextual retrieval (chunk contextualization before embedding — high value, medium effort).

---

## Stage 6 — Reranking

**Implementation:** `knowledge_base.py:444-464` — single `term_overlap` reranker: `0.55·retrieval_score + 0.30·term_overlap + 0.10·bigram_overlap + 0.05·exact_phrase`. `none` passes through.

**Honest assessment:** this is a **lexical re-scoring heuristic, not a reranker** in the modern sense. The README's "lightweight reranking" wording is accurate; anything implying cross-encoder quality would be misleading. Mixing RRF scores (≈0.016) with overlap terms (0..1) means the `0.55·retrieval_score` term is nearly meaningless after fusion — the heuristic is effectively dominated by lexical overlap.

**Upgrade:** **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~70 MB, CPU-capable) behind a config flag with the current heuristic retained as the lightweight default — high value, medium effort, must run *outside* the global lock with top-n candidate restriction (e.g., rerank top 20 → top 5).

---

## Stage 7 — Context Construction

**Implementation:** top-k chunks → `ContextChunk` list (text, score, metadata) + parallel `Citation` list (label = filename + page/chunk). The same ordered list feeds the LLM prompt (`SOURCE [i]` blocks) and the API response — indices align by construction (verified in code).

**Limitations:** no context compression/dedup (overlapping chunks can repeat passages); no per-chunk trust levels; page markers leak into context text; `top_k ≤ 20` but no token-budget-aware context assembly (a 20×800-token context can blow the LLM budget with no guard).

**Upgrade:** token-budgeted context assembly + near-duplicate chunk collapse (low effort, medium value); contextual compression via LLM summarization of chunks (medium effort — defer).

---

## Stage 8 — LLM Response Generation

**Implementation:** `llm.py` — raw `urllib` POST to an OpenAI-compatible `/chat/completions`, temperature 0.1, strict grounding system prompt, inline `[i]` citation instruction. `enabled` requires a non-extractive provider **and** an API key.

**Verified:** works structurally; extractive fallback (sentence-overlap scoring, max 4 sentences, `[i]` suffixes) is tested and deterministic. **The LLM HTTP path has zero test coverage** (no mock), single attempt (no retry/backoff), no streaming, no token counting, no provider abstraction beyond a base-URL string, error details swallowed into `LLMError("LLM provider request failed.")` (loses HTTP status — a 401 vs 429 vs 500 are indistinguishable to operators).

**Limitations vs. modern practice:** no streaming (UI waits for full answer); no function/tool use; no structured output; no prompt-injection defenses beyond the system prompt (document text is inserted raw between `Context:` and `Available sources:` — an injected instruction inside a document sits in the *user* turn alongside the question); `RAG_LLM_PROVIDER` is essentially a display string (any value ≠ `extractive` + key = enabled).

**Upgrade options:** streaming SSE (high value, medium effort — native `StreamingResponse`) → provider adapters (OpenAI/Anthropic/Gemini/Ollama) with retry-on-429/5xx (high value, medium) → token-budget enforcement (low effort) → injection-hardened prompt packaging (see Security report — high value, medium).

---

## Stage 9 — Citations, Grounding & Hallucination Control

**Implementation:** `api.py:269-373` — citations built from retrieved chunks; `grounded=True` set whenever generation runs; knowledge-gap response when evidence is weak; heuristic outdated/conflicting-source warnings.

**Verified:** knowledge-gap gating tested both directions (gap=True for absent info; grounded=True with citations for present info). Confidence is hand-tuned arithmetic (`api.py:284-295`) with different scales for hybrid (RRF) vs cosine retrieval.

**Honest assessment — what is real vs. claimed:**
- ✅ **Real:** retrieval-grounded context, extractive answers are grounded by construction, explicit no-answer behavior, source labels, warnings.
- ⚠️ **Asserted, not proven:** `grounded=True` for LLM answers — **no citation verification exists** (nothing checks that the answer's claims/`[i]` markers actually correspond to the cited chunks), no NLI/entailment, no faithfulness score.
- ⚠️ **Heuristic:** confidence values are not calibrated (no empirical grounding; hybrid single-chunk = fixed 0.72); conflicting-source detection compares number-sets across chunks (false positives on shared tables/units likely).

**Upgrade options:** citation-precision verification (every `[i]` must exist and the sentence must lexically/NLI-entail the cited chunk — high value, medium effort) → LLM-as-judge faithfulness scoring in evals (see `RAG_EVALUATION_STRATEGY.md`) → calibrated confidence (fit thresholds against eval data — later phase).

---

## Stage 10 — Evaluation

**Implementation:** `evals.py` + `POST /evals/run` — retrieval-only metrics: **hit rate** and **MRR** over expected source IDs, per-case trace.

**Verified:** endpoint and unit tests pass (with numpy).

**Limitations:** no generation metrics (faithfulness, answer relevance, hallucination rate), no no-answer/abstention accuracy metric (despite the system's flagship knowledge-gap feature!), no citation correctness, no latency tracking, no golden dataset shipped in the repo, no CI integration, no regression thresholds. The README is *honest* about this ("focuses on retrieval metrics") — but a production-quality platform needs the full suite. Details in `RAG_EVALUATION_STRATEGY.md`.

---

## Stage 11 — Query Latency & Resource Profile (measured)

| Scenario | Result |
|---|---|
| TF-IDF query @ 2,000 chunks | **94.6 ms** (linear scan, pure Python) |
| Hybrid query @ 2,000 chunks | **199.9 ms** (2 legs + RRF + rerank) |
| Ingest 500 docs (2,000 chunks) | **15.42 s** (O(N²) doc-freq rebuilds) |
| Single ingest @ 2,000 chunks | **76.6 ms** (full rebuild + full JSON rewrite) |
| First semantic query after restart | Full corpus re-embed (unbounded; model load under lock) |
| LLM generation | +30 s worst case (urllib timeout), no streaming |

**Projection:** 10,000 chunks → ~0.5 s TF-IDF queries, ~2 s hybrid, ~400 ms+ per ingest. The design ceiling is ~10–20k chunks single-process — fine for a portfolio deployment, insufficient for production.

---

## Summary — Pipeline Maturity by Stage

| Stage | Maturity | One-line verdict |
|---|---|---|
| Ingestion/parsing | 6/10 | Solid basics; no OCR/tables/versioning; query-param validation gaps |
| Chunking | 7/10 | Genuinely good strategies; no structure/parent-child awareness |
| Embeddings | 5/10 | Great abstraction, broken default, nothing persisted |
| Storage | 3/10 | Correct but JSON-bound; the #1 scaling constraint |
| Retrieval | 6/10 | Correct RRF hybrid; slow linear scans; no BM25/ANN/multi-query |
| Reranking | 3/10 | Lexical heuristic mislabeled as reranking |
| Context construction | 5/10 | Aligned citations; no compression or token budgeting |
| Generation | 4/10 | Works; untested, no streaming/retries/provider abstraction |
| Grounding/citations | 6/10 | Real gating + citations; `grounded` asserted, not verified |
| Evaluation | 4/10 | Retrieval metrics only; no generation/abstention/citation eval |
| Overall pipeline | **5/10** | Honest, debuggable 2023-era hybrid RAG — strong base to upgrade |

**The three highest-leverage pipeline changes:**
1. **Persist vectors in a real vector store (Qdrant)** — fixes storage, latency, and re-embedding in one move.
2. **Fix defaults + reranking** (dependency-free default for dev/CI, MiniLM default for production, cross-encoder option) — fixes first-run experience and retrieval quality.
3. **Add citation verification + LLM-as-judge evals in CI** — turns "grounded" from a claim into a measured property.
