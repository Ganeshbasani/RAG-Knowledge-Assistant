from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    provider: str
    model: str | None
    latency_ms: float
    used_llm: bool


class LLMError(RuntimeError):
    """Raised when an external LLM provider cannot generate a response."""


class AnswerGenerator:
    """Grounded answer generator with a no-key extractive fallback."""

    def __init__(self, provider: str = "extractive", model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1/chat/completions", timeout_seconds: int = 30) -> None:
        self.provider = provider.strip().lower()
        self.model = model.strip() if model else None
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(5, timeout_seconds)

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "disabled", "none", "extractive"} and bool(self.api_key)

    @staticmethod
    def _sentences(context: list[dict[str, Any]]) -> list[str]:
        sentences: list[str] = []
        for item in context:
            for sentence in re.split(r"(?<=[.!?])\s+", item.get("text", "").strip()):
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
        return sentences

    @staticmethod
    def extractive(question: str, context: list[dict[str, Any]], max_sentences: int = 4) -> str:
        question_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        scored: list[tuple[float, str, int]] = []
        for source_index, item in enumerate(context, start=1):
            for sentence in re.split(r"(?<=[.!?])\s+", item.get("text", "").strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                overlap = len(question_terms & terms) / max(1, len(question_terms))
                length_penalty = min(len(sentence) / 500.0, 1.0)
                score = overlap + (0.15 * (1.0 - length_penalty))
                if overlap > 0:
                    scored.append((score, sentence, source_index))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, sentence, source_index in scored:
            key = sentence.lower()
            if key in seen:
                continue
            selected.append(f"{sentence} [{source_index}]")
            seen.add(key)
            if len(selected) >= max_sentences:
                break
        if not selected:
            return "I couldn't find sufficient evidence in the indexed documents to answer this question."
        return " ".join(selected)

    def generate(self, question: str, context: list[dict[str, Any]], citations: list[dict[str, Any]]) -> GenerationResult:
        started = time.perf_counter()
        if not context:
            return GenerationResult(
                answer="I couldn't find sufficient evidence in the indexed documents to answer this question.",
                provider="none",
                model=None,
                latency_ms=0.0,
                used_llm=False,
            )

        if not self.enabled:
            answer = self.extractive(question, context)
            return GenerationResult(
                answer=answer,
                provider="extractive",
                model=None,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                used_llm=False,
            )

        citation_text = "\n".join(
            f"[{idx}] {item['source_id']} — chunk {item['chunk_index']}"
            for idx, item in enumerate(citations, start=1)
        )
        context_text = "\n\n".join(
            f"SOURCE [{idx}]\n{item['text']}"
            for idx, item in enumerate(context, start=1)
        )
        system = (
            "You are a grounded knowledge assistant. Answer only from the provided source context. "
            "Do not invent facts. If the context does not contain enough evidence, say so explicitly. "
            "Use inline citations like [1] and [2] that correspond to the source list. Keep the answer concise."
        )
        user = (
            f"Question:\n{question}\n\nContext:\n{context_text}\n\nAvailable sources:\n{citation_text}\n\n"
            "Return a helpful answer grounded only in the context."
        )
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = str(result["choices"][0]["message"]["content"]).strip()
            if not answer:
                raise LLMError("LLM returned an empty answer.")
            return GenerationResult(
                answer=answer,
                provider=self.provider,
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                used_llm=True,
            )
        except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError("LLM provider request failed.") from exc
