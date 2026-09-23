# Architecture Notes

## Request flow

1. The client uploads a supported document or sends raw text.
2. The document processor extracts text and adds stable source metadata.
3. The knowledge base chunks the text and performs an idempotent source upsert.
4. Query requests run keyword, semantic, or hybrid retrieval.
5. Hybrid retrieval fuses ranked candidates using Reciprocal Rank Fusion.
6. The reranker performs a second relevance pass.
7. The evidence gate checks whether useful evidence exists.
8. If evidence is weak, the assistant returns a knowledge-gap response instead of inventing an answer.
9. Otherwise the generation layer produces a grounded answer and source citations. Without an LLM key, an extractive fallback is used.
10. Runtime metrics and request IDs make requests easier to debug.

## Design decisions

### Why hybrid retrieval?

Keyword retrieval is useful for exact product names, identifiers, error messages, and terminology. Semantic retrieval helps with conceptual similarity. Combining them makes the small reference implementation more robust.

### Why source upsert?

Re-indexing the same document should replace the current source rather than create duplicate chunks.

### Why confidence-aware no-answer behavior?

A retrieval system should distinguish between "I found evidence" and "I found some text." Returning an explicit knowledge gap is safer than passing weak context to a generator.

### Why an extractive fallback?

The repository should remain runnable for development and portfolio demonstrations without forcing every user to configure a paid external LLM provider.

## Scaling boundary

The local JSON index is intentionally simple and easy to understand. It is not intended for horizontal multi-worker scale. A larger deployment should move chunks, document metadata, and embeddings to shared persistence such as PostgreSQL + pgvector or a managed vector database, and move indexing to background workers.
