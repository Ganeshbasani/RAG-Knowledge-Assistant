# TrustAware RAG Knowledge Assistant

A production-style RAG application for document-grounded Q&A. Users upload documents, ask questions, and receive evidence-backed answers with source citations, confidence checks, and diagnostics for missing, stale, or potentially conflicting information.

## Core capabilities

- PDF, DOCX, TXT, and Markdown ingestion
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

Without an external key, the app uses a deterministic extractive answerer so the project remains runnable.

For an OpenAI-compatible provider:

```env
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_API_KEY=your-key
RAG_LLM_MODEL=gpt-4o-mini
RAG_LLM_BASE_URL=https://api.openai.com/v1/chat/completions
```

You may point `RAG_LLM_BASE_URL` to another compatible provider.

## Example query

```json
{
  "question": "What is the remote work policy?",
  "retrieval": "hybrid",
  "embedding_provider": "local_tfidf",
  "reranker": "term_overlap",
  "top_k": 5,
  "generate": true
}
```

The response includes `answer`, `grounded`, `confidence`, `knowledge_gap`, `citations`, `warnings`, `context`, `generation`, and `trace`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness / storage probe |
| GET | `/stats` | Index statistics |
| GET | `/documents` | List documents |
| POST | `/ingest` | Ingest raw text |
| POST | `/ingest/file` | Upload PDF/DOCX/TXT/MD |
| POST | `/ingest/bulk` | Bulk ingestion |
| POST | `/query` | Grounded query |
| POST | `/query/semantic` | Semantic-only query |
| DELETE | `/documents/{source_id}` | Delete a source |
| DELETE | `/clear` | Clear index |
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
│   └── models.py
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
