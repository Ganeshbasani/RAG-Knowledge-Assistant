"""Unit tests for LLM provider adapters (Phase 2.2) — transports mocked."""

from __future__ import annotations

import json
from unittest import mock
from urllib import error

import pytest

from rag_assistant.llm import (
    AnthropicAdapter,
    GeminiAdapter,
    LLMError,
    OllamaAdapter,
    OpenAICompatibleAdapter,
)


class _FakeResponse:
    def __init__(self, payload: dict, lines: list[bytes] | None = None):
        self._payload = payload
        self._lines = lines or []

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _openai_payload(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_openai_adapter_generate_parses_response():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x")
    with mock.patch("rag_assistant.llm.request.urlopen", return_value=_FakeResponse(_openai_payload("hello world"))):
        assert adapter.generate("q", [], []) == "hello world"


def test_openai_adapter_generate_retries_on_429():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_retries=2, timeout_seconds=5)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise error.HTTPError("url", 429, "rate limited", {}, None)
        return _FakeResponse(_openai_payload("ok"))

    with mock.patch("rag_assistant.llm.request.urlopen", side_effect=fake_urlopen), \
            mock.patch("rag_assistant.llm._backoff_sleep"):
        assert adapter.generate("q", [], []) == "ok"
    assert calls["n"] == 2


def test_openai_adapter_generate_raises_after_retries_exhausted():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_retries=1)
    with mock.patch("rag_assistant.llm.request.urlopen", side_effect=error.URLError("boom")), \
            mock.patch("rag_assistant.llm._backoff_sleep"):
        with pytest.raises(LLMError):
            adapter.generate("q", [], [])


def test_openai_adapter_generate_stream_yields_deltas():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x")
    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    with mock.patch("rag_assistant.llm.request.urlopen", return_value=_FakeResponse({}, lines=sse_lines)):
        text = "".join(adapter.generate_stream("q", [], []))
    assert text == "Hello world"


def test_openai_adapter_output_token_budget():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_output_tokens=2)
    with mock.patch("rag_assistant.llm.request.urlopen", return_value=_FakeResponse(_openai_payload("x" * 100))):
        assert len(adapter.generate("q", [], [])) <= 8


def test_anthropic_adapter_generate():
    adapter = AnthropicAdapter("claude-x", "key")
    payload = {"content": [{"type": "text", "text": "hi there"}]}
    with mock.patch("rag_assistant.llm.request.urlopen", return_value=_FakeResponse(payload)):
        assert adapter.generate("q", [], []) == "hi there"


def test_gemini_adapter_generate():
    adapter = GeminiAdapter("gem-x", "key")
    payload = {"candidates": [{"content": {"parts": [{"text": "gem answer"}]}}]}
    with mock.patch("rag_assistant.llm.request.urlopen", return_value=_FakeResponse(payload)):
        assert adapter.generate("q", [], []) == "gem answer"


def test_ollama_adapter_generate():
    adapter = OllamaAdapter("llama3.1:8b")
    payload = {"message": {"content": "ollama answer"}}
    with mock.patch("rag_assistant.llm.request.urlopen", return_value=_FakeResponse(payload)):
        assert adapter.generate("q", [], []) == "ollama answer"
