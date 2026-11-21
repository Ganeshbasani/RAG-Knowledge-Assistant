# Roadmap

## Near term (Phase 4)

- Per-key/workspace rate limiting (4.2) � replace pure-IP keying.
- Structured JSON logs with `request_id` correlation + Prometheus metrics (4.3).

## Near term


- Increase observability: latency percentiles and retrieval stage timings in `/metrics`.
- Add per-workspace metadata isolation with server-enforced filters (JWT + workspace claims).
- Add production hardening: audit log for destructive operations, upload parse limits.

## Mid term

- Async ingestion with retry policies and operator dashboards.
- LLM-as-judge faithfulness/relevance scoring (temperature 0, JSON output) — new nightly CI job.
- Calibrate confidence thresholds from eval-labeled data (fit logistic/isotonic on retrieval features).

## Long term

- True multi-tenant workspaces (isolated retrieval by tenant).
- Sharded vector index support and export/import tooling.
- Client-side feedback loop → re-embed + eval datasets (see `RAG_EVALUATION_STRATEGY.md`).
