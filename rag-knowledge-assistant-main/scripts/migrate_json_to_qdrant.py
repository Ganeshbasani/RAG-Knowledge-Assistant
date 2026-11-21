"""Migrate the JSON chunk index to a Qdrant collection, with verification.

Reads the legacy JSON engine (``JSONVectorStore``), re-embeds vectors through
the same ``KnowledgeBase``-level provider flow, upserts chunks + vectors into
Qdrant, then verifies counts and retrieval parity (sample queries compared
between the JSON engine and Qdrant).

Usage:
    python scripts/migrate_json_to_qdrant.py [--storage-path data/rag-index.json]
                                              [--qdrant-url http://localhost:6333]
                                              [--collection rag-chunks]
                                              [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rag_assistant.stores import JSONVectorStore, QdrantVectorStore
from rag_assistant.knowledge_base import KnowledgeBase


def migrate(storage_path: str, qdrant_url: str, collection: str, *, dry_run: bool = False) -> int:
    json_store = JSONVectorStore(storage_path)
    chunks = json_store.load_chunks()
    vectors = json_store.load_vectors()
    print(f"source index: {storage_path}")
    print(f"  chunks: {json_store.count()}")
    print(f"  vectors: {len(vectors)}")

    # Reference queries through the JSON engine for parity comparison.
    kb = KnowledgeBase(vector_store=json_store, storage_path=storage_path or None)
    questions = [
        "How many remote days are allowed?",
        "What is the remote work policy?",
        "budget approval process manager sign-off",
        "conflicting information in documents",
    ]
    for question in questions:
        chunks_out, _ = kb.query_with_trace(question, top_k=5, min_score=0.08, retrieval="hybrid",
                                            embedding_provider="local", reranker="none")
        print(f"  reference query {question!r}: {[(c.source_id, c.chunk_index) for c in chunks_out]}")

    if dry_run:
        print("DRY RUN: no writes performed.")
        return 0

    qdrant = QdrantVectorStore(url=qdrant_url, collection=collection)
    qdrant.save_chunks(chunks)
    if vectors:
        qdrant.save_vectors(vectors, embedding_key=json_store.load_vector_key())
    print(f"target: qdrant://{qdrant_url}/{collection}")
    print(f"  chunks after migration: {qdrant.count()}")
    print(f"  vectors after migration: {len(qdrant.load_vectors())}")

    # Parity check: every migrated chunk and vector is present in Qdrant.
    migrated_chunks = {(c["source_id"], c["chunk_index"]) for c in qdrant.load_chunks()}
    migrated_vectors = qdrant.load_vectors()
    missing_chunks = [(c["source_id"], c["chunk_index"]) for c in chunks
                      if (c["source_id"], c["chunk_index"]) not in migrated_chunks]
    missing_vectors = [key for key in vectors if key not in migrated_vectors]
    if missing_chunks or missing_vectors:
        print(f"  parity FAILED; missing chunks: {missing_chunks[:5]}, missing vectors: {missing_vectors[:5]}")
        return 1
    print(f"  parity: {len(chunks)}/{len(chunks)} chunks present in Qdrant")
    print("MIGRATION COMPLETE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate the JSON index to Qdrant.")
    parser.add_argument("--storage-path", default="data/rag-index.json")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="rag-chunks")
    parser.add_argument("--dry-run", action="store_true", help="Verify and report without writing to Qdrant.")
    args = parser.parse_args()
    return migrate(args.storage_path, args.qdrant_url, args.collection, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
