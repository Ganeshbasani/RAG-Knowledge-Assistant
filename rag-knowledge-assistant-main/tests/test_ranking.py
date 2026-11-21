from __future__ import annotations

from rag_assistant.ranking import BM25Index


def test_bm25_exact_match_scores_highest():
    index = BM25Index()
    index.add(("a", 0), ["apple", "banana", "cherry"])
    index.add(("a", 1), ["apple", "apple", "banana", "durian", "elderberry", "fig"])
    index.add(("a", 2), ["zebra", "yak", "xylophone"])
    results = dict(index.top_k(["apple"], k=10))
    assert results[("a", 0)] > 0
    assert results[("a", 1)] > 0
    assert index.top_k(["zebra"], k=10)[0][0] == ("a", 2)


def test_bm25_top_k_respects_limit():
    index = BM25Index()
    for i in range(10):
        index.add(("s", i), [f"term{i}"])
    assert len(index.top_k(["term1"], k=3)) == 1
    assert index.top_k(["term1"], k=0) == []


def test_bm25_no_match_returns_empty():
    index = BM25Index()
    index.add(("a", 0), ["alpha", "beta"])
    assert index.top_k(["zzz-missing"], k=5) == []


def test_bm25_remove_and_remove_source():
    index = BM25Index()
    index.add(("a", 0), ["alpha", "beta"])
    index.add(("a", 1), ["gamma", "delta"])
    index.add(("b", 0), ["alpha", "omega"])
    assert len(index) == 3
    index.remove(("a", 0))
    assert len(index) == 2
    assert ("a", 0) not in dict(index.top_k(["alpha"], k=10))
    assert index.remove_source("a") == 1
    assert len(index) == 1
    assert ("b", 0) in dict(index.top_k(["alpha"], k=10))
    assert index.remove_source("missing") == 0


def test_bm25_rebuild():
    index = BM25Index()
    index.add(("a", 0), ["alpha", "beta"])
    index.add(("b", 0), ["beta", "gamma"])
    index.rebuild({("c", 0): ["gamma", "delta"]})
    assert len(index) == 1
    assert set(dict(index.top_k(["gamma"], k=5))) == {("c", 0)}


def test_bm25_empty_corpus():
    index = BM25Index()
    assert index.top_k(["anything"], k=5) == []
