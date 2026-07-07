
#  RAG Knowledge Assistant
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&center=true&vCenter=true&width=900&lines=AI-Powered+RAG+Knowledge+Assistant;Hybrid+Search+%7C+Semantic+Retrieval;FastAPI+%7C+Vector+Ready+%7C+Production+Ready;+by+Ganesh+Basani" alt="Typing SVG" />
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/RAG-Hybrid%20Retrieval-blueviolet?style=for-the-badge"/>
</p>

---

## ✨ Overview

RAG Knowledge Assistant is a production-ready Retrieval-Augmented Generation backend designed for scalable document intelligence.

It combines lexical search, semantic embeddings, hybrid retrieval, reranking, evaluation tools, persistent storage, observability, and secure APIs into one developer-friendly platform.

---

# 🌟 Highlights

- Smart Document Chunking
- Hybrid Retrieval
- TF-IDF + Semantic Search
- Reciprocal Rank Fusion
- Query Tracing
- Evaluation Harness
- API Key Authentication
- Rate Limiting
- Docker Ready
- Render + Vercel Deployment
- Metrics & Health Checks

---

# 🏗 Architecture

```text
                User
                  │
          FastAPI REST API
                  │
      ┌───────────┴────────────┐
      │                        │
Document Ingestion       Query Engine
      │                        │
 Smart Chunking        Hybrid Retrieval
      │                        │
 TF-IDF + Embeddings + Reranker
                  │
          Ranked Context
                  │
             Final Response
```

---

# ⚡ Features

| Feature | Description |
|----------|-------------|
| Smart Chunking | Token, Sentence, Paragraph & AI-aware |
| Hybrid Retrieval | TF-IDF + Semantic |
| Semantic Providers | sentence-transformers, local, ONNX |
| Query Trace | Debug retrieval pipeline |
| Evaluation | Offline benchmarking |
| Security | API Keys + Validation |
| Monitoring | Metrics + Health endpoints |
| Deployment | Docker, Render, Vercel |

---

# 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/rag-knowledge-assistant.git

cd rag-knowledge-assistant

python -m venv .venv

source .venv/bin/activate

pip install -e ".[dev]"
```

Run

```bash
cp .env.example .env

uvicorn rag_assistant.api:app --reload --app-dir src
```

Open

```
http://localhost:8000/docs
```

---

# 📂 Project Structure

```text
rag-knowledge-assistant
│
├── src/
├── ui/
├── tests/
├── docs/
├── render.yaml
├── vercel.json
├── README.md
└── LICENSE
```

---

# 🔎 API

| Method | Endpoint |
|---------|----------|
| GET | /health |
| GET | /ready |
| GET | /metrics |
| POST | /ingest |
| POST | /query |
| POST | /query/semantic |
| POST | /evals/run |
| DELETE | /documents/{source_id} |

---

# 🔐 Environment Variables

```env
RAG_API_KEY=
RAG_STORAGE_PATH=
RAG_HOST=0.0.0.0
RAG_PORT=8000
RAG_ALLOWED_ORIGINS=
```

---

# 🐳 Docker

```bash
docker compose up --build
```

---

# ☁ Deployment

## Render

- Deploy with Docker
- Configure environment variables
- Verify `/health`

## Vercel

- Deploy `ui/`
- Connect backend URL
- Configure CORS

---

# 🛣 Roadmap

- Vector DB integrations
- Streaming responses
- Multi-user workspaces
- LLM provider plugins
- Advanced rerankers
- Distributed indexing

---

# 🤝 Contributing

1. Fork
2. Create branch
3. Commit
4. Push
5. Open PR

---

# 📄 License

MIT License

---

<p align="center">

### ⭐ If this project helped you, please leave a star!

Made  by **Ganesh Basani**

</p>
