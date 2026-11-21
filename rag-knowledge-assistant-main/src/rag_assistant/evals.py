from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rag_assistant.knowledge_base import KnowledgeBase


def run_retrieval_eval(kb: KnowledgeBase, cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    total_hits = 0
    total_reciprocal_rank = 0.0

    for case in cases:
        expected_source_ids = [str(item).strip() for item in case.get("expected_source_ids", []) if str(item).strip()]
        chunks, trace = kb.query_with_trace(
            question=case["question"],
            top_k=int(case.get("top_k", 5)),
            min_score=float(case.get("min_score", 0.0)),
            retrieval=str(case.get("retrieval", "hybrid")),
            embedding_model=case.get("embedding_model"),
            embedding_provider=str(case.get("embedding_provider", "local")),
            local_dimensions=int(case.get("local_dimensions", 256)),
            reranker=str(case.get("reranker", "term_overlap")),
            candidate_pool_size=int(case.get("candidate_pool_size", 10)),
        )

        returned_source_ids: list[str] = []
        seen_sources: set[str] = set()
        for chunk in chunks:
            if chunk.source_id in seen_sources:
                continue
            returned_source_ids.append(chunk.source_id)
            seen_sources.add(chunk.source_id)

        match_rank = None
        expected = set(expected_source_ids)
        for index, source_id in enumerate(returned_source_ids, start=1):
            if source_id in expected:
                match_rank = index
                break

        matched = match_rank is not None
        reciprocal_rank = round(1.0 / match_rank, 4) if match_rank is not None else 0.0
        total_hits += int(matched)
        total_reciprocal_rank += reciprocal_rank

        results.append(
            {
                "question": case["question"],
                "expected_source_ids": expected_source_ids,
                "returned_source_ids": returned_source_ids,
                "matched": matched,
                "reciprocal_rank": reciprocal_rank,
                "top_hit_source_id": returned_source_ids[0] if returned_source_ids else None,
                "trace": trace,
            }
        )

    case_count = len(results)
    if case_count == 0:
        return {"cases": 0, "hits": 0, "hit_rate": 0.0, "mrr": 0.0, "results": []}

    return {
        "cases": case_count,
        "hits": total_hits,
        "hit_rate": round(total_hits / case_count, 4),
        "mrr": round(total_reciprocal_rank / case_count, 4),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Golden-set loading (3.3) + extended metrics (3.4)
# ---------------------------------------------------------------------------

def load_cases(path: str) -> list[dict[str, Any]]:
    """Load golden eval cases from a JSONL file."""
    import json

    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _eligible(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r["expected_source_ids"]]


def recall_at_k(results: list[dict[str, Any]], k: int = 5) -> float:
    """Fraction of cases (with expectations) where an expected source is in the top-k."""
    eligible = _eligible(results)
    if not eligible:
        return 0.0
    hits = sum(1 for r in eligible if set(r["expected_source_ids"]) & set(r["returned_source_ids"][:k]))
    return round(hits / len(eligible), 4)


def ndcg(results: list[dict[str, Any]], k: int = 5) -> float:
    """Mean normalized DCG: relevance of returned sources against the expected set."""
    eligible = _eligible(results)
    if not eligible:
        return 0.0
    import math

    total = 0.0
    for r in eligible:
        expected = set(r["expected_source_ids"])
        dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(r["returned_source_ids"][:k]) if sid in expected)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected), k)))
        if idcg:
            total += dcg / idcg
    return round(total / len(eligible), 4)


def abstention_accuracy(results: list[dict[str, Any]]) -> float:
    """Fraction of unanswerable cases (no expected sources) where nothing was retrieved."""
    unanswerable = [r for r in results if not r["expected_source_ids"]]
    if not unanswerable:
        return 1.0
    correct = sum(1 for r in unanswerable if not r["returned_source_ids"])
    return round(correct / len(unanswerable), 4)


def citation_precision(results: list[dict[str, Any]], min_relevance: float = 0.3) -> float:
    """Fraction of cases where the expected (cited) sources are sufficiently represented."""
    eligible = _eligible(results)
    if not eligible:
        return 0.0
    supported = 0
    for r in eligible:
        expected = set(r["expected_source_ids"])
        returned = set(r["returned_source_ids"])
        if expected and len(expected & returned) / max(1, len(expected)) >= min_relevance:
            supported += 1
    return round(supported / len(eligible), 4)
