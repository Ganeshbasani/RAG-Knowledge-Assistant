# RAG Evaluation Strategy — From Retrieval Metrics to Trustworthy Answers

**Audit date:** 2026-11-19
**Status:** PROPOSED — planning only.
**Context:** the current harness (`evals.py`, `POST /evals/run`) measures **retrieval hit rate and MRR only** — verified working. This document defines the full evaluation system needed to claim (and protect) answer quality.

---

## 1. What Exists Today (verified)

| Capability | Status |
|---|---|
| Hit rate @k over expected `source_id`s | ✅ implemented & tested |
| MRR over expected sources | ✅ implemented & tested |
| Per-case retrieval trace | ✅ returned in results |
| Golden dataset in repo | ❌ none |
| Generation metrics | ❌ none |
| Abstention (knowledge-gap) accuracy | ❌ none — despite being the flagship feature |
| Citation correctness | ❌ none |
| Latency tracking in evals | ❌ none |
| CI integration / regression gates | ❌ none (CI is currently red for other reasons) |

**Key insight:** the system's differentiator is *trust* (evidence gate, citations, knowledge-gap). An evaluation strategy that ignores abstention and citations fails to measure the very properties the product claims.

---

## 2. Evaluation Framework — Four Layers

```
Layer 1  Retrieval quality      (is the right evidence found?)
Layer 2  Generation quality     (is the answer faithful & relevant?)
Layer 3  Trust behaviors        (does it abstain/cite correctly?)
Layer 4  System qualities       (latency, cost, robustness)
```

### Layer 1 — Retrieval metrics (extend existing harness)

| Metric | Definition | Current |
|---|---|---|
| Hit Rate@k | fraction of queries with ≥1 expected source in top-k | ✅ |
| MRR | mean reciprocal rank of first expected source | ✅ |
| Recall@k / nDCG@k | graded relevance (add optional `relevance_grade` per doc) | ➕ |
| Context precision/recall | chunk-level (not just source-level) relevance | ➕ |
| No-evidence detection | queries with *no* relevant doc must return 0 chunks above gate | ➕ |

### Layer 2 — Generation metrics (LLM-as-judge using the existing LLM client)

| Metric | Method | Notes |
|---|---|---|
| **Faithfulness** | decompose answer into claims; each claim must be entailed by cited context (judge prompt or NLI model) | the core hallucination measure |
| **Answer relevance** | judge: does the answer address the question given context? | 1–5 scale, averaged |
| **Citation precision/recall** | every `[i]` marker must (a) exist, (b) point to a chunk that supports the sentence | cheap lexical + judge hybrid; automatable |
| **Answer correctness** | compare against reference answers where available | needs gold answers in dataset |

**Judge configuration:** temperature 0, JSON-structured output, **judge model ≠ generator model** where possible (avoid self-preference bias), 10% human audit of judge judgments (report judge-human agreement Cohen's κ).

*Fallback when no LLM key:* run Layer 2 in "offline mode" with a small local model (e.g., via Ollama) or skip gracefully — the harness must always produce Layer 1 + Layer 3 + Layer 4 results without any LLM (keeps CI free of API costs).

### Layer 3 — Trust behaviors (the differentiators)

| Test | Method |
|---|---|
| **Abstention accuracy** | dataset of unanswerable questions → expect `knowledge_gap=True`, `grounded=False`; measure false-answer rate |
| **False-knowledge-gap rate** | answerable questions wrongly gated → measures over-conservatism |
| **Citation verification rate** | automated check: cited chunk contains supporting sentence (entailment) |
| **Prompt-injection resistance** | adversarial documents containing *"ignore instructions / reveal system prompt"* → assert answer doesn't comply; run as a dedicated eval suite |
| **Stale/conflict warning accuracy** | seeded conflicting docs → expect warnings fired (currently heuristic, untested) |

### Layer 4 — System qualities

| Metric | Target (initial SLO proposal) |
|---|---|
| p50 / p95 query latency (retrieval-only) | <100 ms / <300 ms @10k chunks (post-upgrade) |
| p95 end-to-end with LLM | <5 s (streaming: first token <1.5 s) |
| Ingest throughput | ≥1 doc/s background (post-upgrade) |
| Error rate | <0.5% 5xx excluding deliberate 4xx |
| Eval runtime in CI | <2 min without LLM keys |

---

## 3. Golden Datasets (to be created)

```
tests/fixtures/eval/
├── corpus/                     # small, license-clean documents (~10-20 files)
├── retrieval_cases.jsonl       # {question, expected_source_ids, expected_chunk_ids?, grade}
├── answerable_cases.jsonl      # + {reference_answer, must_contain: [...]}
├── unanswerable_cases.jsonl    # questions the corpus cannot answer
├── adversarial_cases.jsonl     # prompt-injection docs + probe questions
└── conflict_cases.jsonl        # stale/contradictory sources for warning checks
```

**Sizing guidance:** start with 50 retrieval + 30 answerable + 20 unanswerable + 10 adversarial + 10 conflict cases — enough for signal, small enough to hand-verify. Grow to 200+ with graded relevance once the pipeline stabilizes. *Every case is human-verified once;* the dataset is versioned in git.

**Dataset hygiene:** no scraped copyrighted text; questions written *from* the corpus but not lexically trivial (include paraphrases so lexical-only retrieval can't trivially win); balanced across documents.

---

## 4. Harness Design (extends `evals.py`, keeps its API shape)

```python
run_suite(kb, suites) -> {
  "retrieval": {hit_rate, mrr, recall_at_k, ndcg},
  "trust":     {abstention_accuracy, false_gap_rate, citation_precision, injection_pass_rate},
  "generation": {faithfulness, answer_relevance, citation_recall} | {"skipped": "no_llm_key"},
  "system":    {p50_latency_ms, p95_latency_ms, errors},
  "cases":     [ per-case detail with trace ]        # keep existing per-case shape
}
```

- **Offline (CI):** `POST /evals/run` gains `suite` support; CLI command `rag-eval --suite all --no-judge` for the free tier.
- **Regression gate:** store each run's summary (new `eval_results` table/file); CI fails if any tracked metric drops >2 points vs. baseline (allow-listed for first runs).
- **Online (post-launch):** sample 5% of production queries → async eval + user feedback (thumbs) join the same dataset → periodic judge scoring → dashboards.

---

## 5. Confidence Calibration (turns heuristics into measured behavior)

Current confidence (`api.py:284-295`) is hand-tuned arithmetic with retrieval-mode-dependent scales — *code-inspected, untested*.

**Plan:**
1. Log `(confidence, correctness_label)` pairs from eval runs (correctness = judge verdict on answerable; abstention-correct on unanswerable).
2. Fit a simple calibrator (isotonic regression or logistic on features: best score, score separation, chunk count, reranker margin).
3. Choose gate threshold = target **false-answer rate** (e.g., ≤5% of answers ungrounded) and verify on held-out cases.
4. Re-validate calibrator whenever retrieval/rerank stages change (the regression gate enforces this).

Outcome: `confidence` becomes an empirically meaningful number and the UI badge thresholds (`.72 / .45`) get evidence instead of guesses.

---

## 6. CI/CD Integration

```yaml
jobs:
  lint:      ruff + mypy + bandit            # pinned versions (fixes verified drift)
  unit:      pytest + coverage ≥ 80% gate     # existing suite + new tests
  eval-free: rag-eval --no-judge              # Layer 1+3+4, no API cost, <2 min
  eval-llm:  rag-eval --judge (scheduled/nightly, secrets-gated; also on PRs labeled 'rag-change')
  docker:    build + smoke (health, ingest, query against fixture corpus)
```
- Nightly LLM-judge runs catch quality drift without blocking every PR on API spend.
- Baseline JSON committed; deltas reported in the PR comment (tables of metric → old → new).

---

## 7. Implementation Order

| Step | Content | Depends on | Effort |
|---|---|---|---|
| 1 | Fix CI (pinned tools, numpy default issue) — *prerequisite for any gate* | — | 1 day |
| 2 | Golden datasets (corpus + 4 case files, human-verified) | — | 2–3 days |
| 3 | Extend harness: recall@k, nDCG, abstention metrics, latency | datasets | 2 days |
| 4 | Citation verification (lexical → judge) | datasets | 2–3 days |
| 5 | LLM-as-judge faithfulness/relevance + judge agreement audit | LLM adapter w/ retries | 3 days |
| 6 | Confidence calibration + gate threshold selection | 3–5 | 2 days |
| 7 | CI gates + baseline + nightly judge job | 1–3 | 1–2 days |
| 8 | Online feedback loop (thumbs → dataset → periodic eval) | feedback endpoint (roadmap P3) | 2 days |

**Total: ~2–2.5 weeks** — deliberately placed *after* storage/defaults fixes (Phase 0–1 of the roadmap) but *before* advanced features, so every subsequent upgrade is measured rather than guessed.

---

## 8. Success Criteria (definition of done for the evaluation system)

- [ ] CI fails on: lint errors, test failures, coverage <80%, retrieval metric regressions >2 pts.
- [ ] Every public claim maps to a metric: "grounded" → faithfulness ≥0.9; "knowledge-gap" → abstention accuracy ≥0.9; "citations" → citation precision ≥0.95 (thresholds revisited after first baseline run).
- [ ] Adversarial injection suite passes at ≥90% (documented as a known-boundary if lower).
- [ ] Latency SLOs measured in eval runs, not anecdotes.
- [ ] A single command reproduces the full report locally without API keys (judge tier optional).
