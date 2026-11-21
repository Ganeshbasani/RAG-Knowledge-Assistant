"""Reranker implementations.

- ``term_overlap``: the default, lightweight lexical re-scoring heuristic.
- ``cross_encoder``: opt-in real reranking via a sentence-transformers
  cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``, ~70 MB, CPU-capable).
  Requires the ``[embeddings]`` extra installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CrossEncoderReranker:
    """Wraps a ``sentence_transformers.CrossEncoder`` with a lazy load."""

    name = "cross_encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def score_pairs(self, question: str, texts: Sequence[str]) -> list[float]:
        """Return one cross-encoder relevance score per text."""
        if not texts:
            return []
        model = self._load()
        scores = model.predict([(question, text) for text in texts], batch_size=8)
        return [float(score) for score in scores]
