"""Incremental BM25 index for the ``tfidf`` retrieval leg.

The index keeps token→document frequencies, per-document lengths, and the
corpus average length, so ingest/remove are O(tokens-per-chunk) and queries
score over an inverted index instead of re-vectorizing the whole corpus per
request (the previous implementation rebuilt doc-freq from scratch on every
ingest — O(N²) over the corpus).

Scoring uses the standard BM25 formula with smoothed IDF:

    idf = log((N - df + 0.5) / (df + 0.5) + 1)
    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avg_len))
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence


class BM25Index:
    """Incremental BM25 index keyed by ``(source_id, chunk_index)``."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[tuple[str, int], int]] = {}  # token -> key -> tf
        self._key_terms: dict[tuple[str, int], frozenset[str]] = {}
        self._key_source: dict[tuple[str, int], str] = {}
        self._doc_len: dict[tuple[str, int], int] = {}
        self._total_len = 0
        self._avg_len = 0.0

    # ------------------------------------------------------------------ writes
    def rebuild(self, documents: Mapping[tuple[str, int], Sequence[str]]) -> None:
        """Replace the whole index with the given ``key -> tokens`` mapping."""
        self._postings = {}
        self._key_terms = {}
        self._key_source = {}
        self._doc_len = {}
        self._total_len = 0
        self._avg_len = 0.0
        for key, tokens in documents.items():
            self.add(key, tokens)

    def add(self, key: tuple[str, int], tokens: Sequence[str]) -> None:
        source_id, _ = key
        frequencies = Counter(tokens)
        unique_tokens = frozenset(frequencies)
        self._key_terms[key] = unique_tokens
        self._key_source[key] = source_id
        self._doc_len[key] = len(tokens)
        self._total_len += len(tokens)
        self._avg_len = self._total_len / len(self._doc_len) if self._doc_len else 0.0
        for token, tf in frequencies.items():
            posting = self._postings.setdefault(token, {})
            posting[key] = posting.get(key, 0) + tf

    def remove(self, key: tuple[str, int]) -> None:
        tokens = self._key_terms.pop(key, frozenset())
        self._key_source.pop(key, None)
        doc_len = self._doc_len.pop(key, None)
        if doc_len is not None:
            self._total_len -= doc_len
            self._avg_len = self._total_len / len(self._doc_len) if self._doc_len else 0.0
        for token in tokens:
            posting = self._postings.get(token)
            if not posting:
                continue
            posting.pop(key, None)
            if not posting:
                self._postings.pop(token, None)

    def remove_source(self, source_id: str) -> int:
        keys = [key for key, sid in self._key_source.items() if sid == source_id]
        for key in keys:
            self.remove(key)
        return len(keys)

    # ------------------------------------------------------------------- query
    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[tuple[str, int], float]]:
        scores: dict[tuple[str, int], float] = {}
        query_set = set(query_tokens)
        n_docs = len(self._doc_len)
        if n_docs == 0 or self._avg_len == 0:
            return []
        for token in query_set:
            posting = self._postings.get(token)
            if not posting:
                continue
            df = len(posting)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf <= 0:
                continue
            for key, tf in posting.items():
                doc_len = self._doc_len.get(key, 0)
                norm = tf + self._k1 * (1.0 - self._b + self._b * doc_len / self._avg_len)
                if norm <= 0:
                    continue
                scores[key] = scores.get(key, 0.0) + idf * (tf * (self._k1 + 1.0)) / norm
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def __len__(self) -> int:
        return len(self._doc_len)
