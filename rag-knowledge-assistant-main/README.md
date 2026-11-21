# TrustAware RAG Knowledge Assistant

A production-style RAG application for document-grounded Q&A. Users upload documents, ask questions, and receive evidence-backed answers with source citations, confidence checks, and diagnostics for missing, stale, or potentially conflicting information.

## Core capabilities

- PDF, DOCX, PPTX, XLSX, CSV/TSV, JSON, HTML, RTF, TXT, Markdown, and common image ingestion (image text is extracted with RapidOCR)
- Smart chunking with configurable size and overlap
- Keyword, semantic, and hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Lightweight reranking
- Grounded LLM generation through an OpenAI-compatible endpoint
- Safe extractive fallback when no LLM key is configured
- Inline source citations and retrieved evidence
- Confidence-aware no-answer / knowledge-gap handling
- Potential outdated-source and conflicting-information warnings
- Idempotent source replacement and document listing
- Metadata filtering
- API-key authentication and IP-based rate limiting
- Request IDs, health/readiness checks, runtime metrics, and safe errors
- Dockerized deployment with non-root container and health check
- Responsive light UI
- Retrieval evaluation harness

## Architecture

```text
                        Web UI
                           |
                           v
                       FastAPI API
                           |
         +-----------------+-----------------+
         |                 |                 |
         v                 v                 v
  Document Pipeline   Retrieval Engine   Evaluation
         |                 |
  PDF/DOCX/TXT/MD     TF-IDF + Semantic
         |                 |
      Chunking              RRF
         |                 |
     Embeddings          Reranking
         |                 |
         +-----------+-----+
                     v
               Evidence Gate
                 /       \
             enough     weak/none
                |           |
                v           v
               LLM      Knowledge Gap
                |
                v
        Grounded Answer
          + Citations
```

## Quick start

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e ".[dev]"
copy .env.example .env

rag-knowledge-assistant
```

Or:

```bash
uvicorn rag_assistant.api:app --reload --app-dir src
```

Open `http://127.0.0.1:8000/ui/`.

## Enable stronger semantic embeddings

```bash
pip install -e ".[embeddings]"
```

Then choose `sentence_transformers` as the embedding provider and configure `RAG_DEFAULT_EMBEDDING_MODEL` as needed.

## Enable LLM generation

The hosted portfolio setup uses Gemini. Set `RAG_LLM_PROVIDER=gemini`, `RAG_LLM_MODEL=gemini-3.7-flash`, and the server-side `GEMINI_API_KEY`; visitors never provide or see the key. Without a key, the app remains usable with a local extractive fallback.

For an OpenAI-compatible provider instead:

```env
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_API_KEY=your-key
RAG_LLM_MODEL=gpt-6.2-astra
RAG_LLM_BASE_URL=https://api.openai.com/v1/chat/completions
```

You may point `RAG_LLM_BASE_URL` to another compatible provider. ChatGPT subscriptions and API billing are separate; API usage is billed to the OpenAI API project whose key the server uses. See [OpenAI production best practices](https://developers.openai.com/api/docs/guides/production-best-practices).

This setup keeps document upload, indexing, retrieval, and citations in this application and sends only the retrieved context and question to OpenAI for answer generation. To let OpenAI host document search instead, use the [OpenAI File Search tool](https://developers.openai.com/api/docs/guides/tools-file-search); that changes where files are stored and how retrieval is managed.

## Example query

```json
{
  "question": "What is the remote work policy?",
  "retrieval": "hybrid",
  "embedding_provider": "local",
  "reranker": "term_overlap",
  "top_k": 5,
  "generate": true
}
```

The response includes `answer`, `grounded`, `confidence`, `knowledge_gap`, `citations`, `warnings`, `context`, `generation`, and `trace`.

## Authentication and production mode

With no `RAG_API_KEY` set, API authentication is **disabled** and a warning is logged at startup. The public portfolio demo intentionally leaves it unset so visitors can use the UI. For a private deployment, set `RAG_API_KEY` (requests send it as the `x-api-key` header):

```env
RAG_API_KEY=your-long-random-secret
RAG_ENV=production
```

Setting `RAG_ENV=production` without `RAG_API_KEY` refuses to start. Destructive endpoints are step-up protected: `DELETE /clear` requires the `X-Clear-Confirm: DELETE` header in addition to the API key, and `DELETE /documents/{source_id}` returns `404` for unknown sources. Uploaded file queries (`/ingest/file?source_id=…`) are validated with bounds. Requests are capped at 10 MB on the declared `Content-Length` **and** counted as they stream in, so chunked transfers can't bypass the limit.

## Storage & retrieval (Phase 1)

The knowledge base is split into two storage seams (both stable API):

- `VectorStore` — chunk records + embedding vectors.
  - `json` (default): zero-dependency embedded JSON index.
  - `qdrant`: persisted chunks + vectors in Qdrant with payload filters; **Docker Compose ships a Qdrant service + volume** (`docker-compose.yml`).
- `MetadataStore` — documents, feedback, audit events, eval results. SQLite by default (`rag-metadata.db` beside the index), PostgreSQL via `RAG_DB_URL`.

```env
RAG_VECTOR_STORE=json                        # or qdrant
RAG_QDRANT_URL=http://localhost:6333
RAG_QDRANT_API_KEY=
RAG_QDRANT_COLLECTION=rag-chunks
RAG_DB_URL=                                  # e.g. postgresql+psycopg://... (metadata store)
RAG_RERANKER=term_overlap                    # or cross_encoder
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

- Embeddings are persisted (keyed by provider/model/dims) and survive restarts; switching providers invalidates and re-embeds.
- The `tfidf` lexical leg uses an incremental BM25 index (ingest is O(chunk); no more O(N²) corpus rebuilds).
- Cross-encoder reranking (`cross_encoder`) requires the `[embeddings]` extra.

**Migrate an existing JSON index to Qdrant** (the JSON engine is kept as rollback):

```bash
python scripts/migrate_json_to_qdrant.py --storage-path data/rag-index.json \
    --qdrant-url http://localhost:6333 --collection rag-chunks
```

The script upserts chunks + vectors, then verifies counts and that every migrated chunk is present.

## Generation & interaction (Phase 2)

- **Hosted demo model:** Gemini (`gemini-3.7-flash`) is configured server-side. Add `GEMINI_API_KEY` in Render's Environment page; visitors never provide or see the key. Without it, Atlas uses its local extractive fallback. Gemini's free tier has quotas and provider-specific data terms; use public, non-sensitive demo documents only.
- **Free hosting limits:** the Render free service sleeps after inactivity and uses ephemeral disk storage. Uploaded files and the search index may disappear after a restart or redeploy. This setup is suitable for a portfolio demonstration, not a persistent production workspace.

- **Streaming answers:** `POST /query` with `"stream": true` → `text/event-stream` (`delta` events + a final `done` event with the full response). The UI appends tokens to the pending message in real time.
- **Provider adapters:** `RAG_LLM_PROVIDER` defaults to `openai_compatible` with `gpt-6.2-astra` at high reasoning effort for stronger answer quality; supported alternatives include `anthropic`, `gemini`, and `ollama`. Retries with exponential backoff on 429/5xx (3 attempts); optional `RAG_LLM_MAX_OUTPUT_TOKENS` cap.
- **Conversational memory:** `session_id` on `/query` — history (last 10 turns) is injected into the question for retrieval/generation; `GET /sessions` + `DELETE /sessions/{id}` for management.
- **Query rewriting:** LLM-based variants (or rule-based fallback without a key); hybrid runs multi-query retrieval, merged via the existing RRF.
- **Context assembly:** `context_max_tokens` budget, near-duplicate collapse, `[PAGE n]` markers stripped from context text.

## Trust & evaluation (Phase 3)

- **Injection hardening:** untrusted content (question + document excerpts) is wrapped in `<<<UNTRUSTED ... >>>` delimiters; the system prompt carries explicit instruction-priority and data-vs-instructions rules.
- **Citation verification:** for LLM answers, `grounded` is computed — every citation must exist and the cited chunk must share lexical support with the answer (`generation.citation_verification` per response). Extractive answers are grounded by construction.
- **Golden datasets:** `tests/fixtures/eval/` — `retrieval_cases`, `answerable_cases`, `unanswerable_cases`, `adversarial_cases`, `conflict_cases` (hand-verified, versioned in git).
- **Extended harness:** `evals.load_cases()`, `recall_at_k`, `ndcg`, `abstention_accuracy`, `citation_precision` (JSONL loaders for the golden sets).
- **Feedback:** `POST /feedback` / `GET /feedback` — ratings + comments; rows persist in the metadata store and join the eval dataset.
- **Audit log:** `remove_source`/`clear_all` events written to the metadata store; destructive API calls are auditable.
- **Security hardening:** `Content-Security-Policy` on all responses; docs/OpenAPI disabled when an API key is configured.

## Multi-user & production readiness (Phase 4)

- **JWT auth:** `RAG_JWT_SECRET`; `create_token`/`verify_token` (HS256, exp, issuer); `POST /auth/token` to mint tokens; static `RAG_API_KEY` remains the bootstrap admin key.
- **Workspace isolation:** every query/ingest/delete is server-enforced to the caller's `workspace_id` (KB metadata filter, strict on delete); `GET /documents` scoped to the caller; `GET /workspaces`.
- New deps: `pyjwt>=2.10`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness / storage probe |
| GET | `/stats` | Index statistics |
| GET | `/documents` | List documents |
| POST | `/ingest` | Ingest raw text |
| POST | `/ingest/file` | Upload PDF/DOCX/TXT/MD |
| POST | `/ingest/bulk` | Atomic bulk ingestion |
| POST | `/ingest/async` | Queue a document for background ingestion (202 + job id) |
| GET | `/ingest/status/{job_id}` | Background ingestion status |
| GET | `/sessions` | Active session ids (Phase 2.3) |
| DELETE | `/sessions/{session_id}` | Clear a session's history (Phase 2.3) |
| POST | `/query` | Grounded query (add `"stream": true` for SSE) |
| POST | `/query/semantic` | Semantic-only query |
| DELETE | `/documents/{source_id}` | Delete a source (404 when unknown) |
| DELETE | `/clear` | Clear index (requires `X-Clear-Confirm: DELETE`) |
| POST | `/evals/run` | Retrieval evaluation |
| GET | `/metrics` | Runtime metrics |

## Docker

```bash
copy .env.example .env
docker compose up --build
```

The Docker image installs semantic dependencies, runs as a non-root user, and exposes a health check.

## Evaluation

Use a small curated set of representative questions and measure retrieval hit rate, MRR, no-answer accuracy, citation correctness, answer faithfulness/relevance, and response latency. The included evaluation endpoint focuses on retrieval metrics; generation-quality metrics can be layered on top of the returned evidence.

## Project structure

```text
rag-knowledge-assistant/
├── src/rag_assistant/
│   ├── api.py
│   ├── config.py
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── evals.py
│   ├── knowledge_base.py
│   ├── llm.py
│   ├── metadata.py        ← structured metadata store (Phase 1.4)
│   ├── models.py
│   ├── ranking.py         ← BM25 lexical index (Phase 1.6)
│   ├── rerankers.py       ← cross-encoder reranker (Phase 1.7)
│   ├── stores.py          ← VectorStore: JSON + Qdrant (Phase 1)
│   └── __init__.py
├── scripts/
│   └── migrate_json_to_qdrant.py
├── ui/index.html
├── tests/
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── .env.example
└── README.md
```

## Resume-ready project summary

> Built a trust-aware RAG knowledge assistant using Python and FastAPI with document ingestion, hybrid keyword + semantic retrieval, reranking, grounded LLM responses, source citations, confidence-aware no-answer handling, knowledge diagnostics, Docker deployment, and automated retrieval evaluation.

## Important scope note

This release is designed as a **strong production-style portfolio implementation**. The default persistence engine is still a local JSON index, which is suitable for a small deployment and demonstration. For horizontal multi-worker scale, move the index to a shared database/vector store such as PostgreSQL + pgvector or a managed vector database.

## License

MIT
