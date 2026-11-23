"""Provider adapter tests use mocked HTTP transports; no provider calls are made."""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from rag_assistant.llm import (
    AnthropicAdapter,
    GeminiAdapter,
    LLMError,
    OllamaAdapter,
    OpenAICompatibleAdapter,
)


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload


class _FakeStreamResponse(_FakeResponse):
    def __init__(self, lines: list[str], status_code: int = 200, headers: dict | None = None):
        super().__init__({}, status_code, headers)
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(self.lines)


class _FakeHTTPClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def request(self, *args, **kwargs):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, *args, **kwargs):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeLegacyResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self):
        import json

        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _openai_payload(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_openai_adapter_generate_parses_response():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x")
    fake = _FakeHTTPClient([_FakeResponse(_openai_payload("hello world"))])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=fake):
        assert adapter.generate("q", [], []) == "hello world"


def test_openai_adapter_rejects_unexpected_response_shape():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x")
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=_FakeHTTPClient([_FakeResponse({})])):
        with pytest.raises(LLMError) as raised:
            adapter.generate("q", [], [])
    assert raised.value.category == "provider_bad_response"


def test_openrouter_adapter_includes_fallback_model_list():
    adapter = OpenAICompatibleAdapter("openrouter/free", "k", "http://x", fallback_models=("provider/model:free",))
    payload = adapter._messages_payload("system", "question")
    assert payload["models"] == ["provider/model:free"]


def test_openai_adapter_retries_429_and_respects_retry_after():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_retries=2, timeout_seconds=5)
    fake = _FakeHTTPClient([
        _FakeResponse({}, 429, {"Retry-After": "2"}),
        _FakeResponse(_openai_payload("ok")),
    ])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=fake), \
            mock.patch("rag_assistant.llm.time.sleep") as sleep:
        assert adapter.generate("q", [], []) == "ok"
    sleep.assert_called_once_with(2.0)


@pytest.mark.parametrize(("status", "category"), [
    (400, "invalid_request"), (401, "provider_auth"), (402, "provider_payment_required"),
    (403, "provider_forbidden"), (408, "provider_timeout"), (429, "provider_rate_limited"),
    (500, "provider_upstream_error"), (502, "provider_upstream_error"), (503, "provider_unavailable"),
])
def test_openai_adapter_classifies_provider_http_errors(status, category):
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_retries=0)
    fake = _FakeHTTPClient([_FakeResponse({}, status)])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=fake):
        with pytest.raises(LLMError) as raised:
            adapter.generate("q", [], [])
    assert raised.value.category == category
    assert raised.value.status_code == status
    assert "Bearer k" not in str(raised.value)


def test_openai_adapter_bounds_connection_retries():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_retries=1)
    fake = _FakeHTTPClient([httpx.ConnectError("unreachable"), httpx.ConnectError("unreachable")])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=fake), \
            mock.patch("rag_assistant.llm.time.sleep"):
        with pytest.raises(LLMError) as raised:
            adapter.generate("q", [], [])
    assert raised.value.category == "provider_connection_error"


def test_openai_adapter_generates_with_independent_connect_and_read_timeouts():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", connect_timeout_seconds=3, read_timeout_seconds=17)
    timeout = adapter._http_timeout()
    assert timeout.connect == 3
    assert timeout.read == 17


def test_openai_adapter_stream_yields_until_done():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x")
    response = _FakeStreamResponse([
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=_FakeHTTPClient([response])):
        assert "".join(adapter.generate_stream("q", [], [])) == "Hello world"


def test_openai_adapter_rejects_incomplete_stream_instead_of_returning_partial_answer():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_retries=0)
    response = _FakeStreamResponse(['data: {"choices":[{"delta":{"content":"partial"}}]}'])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=_FakeHTTPClient([response])):
        with pytest.raises(LLMError) as raised:
            "".join(adapter.generate_stream("q", [], []))
    assert raised.value.category == "provider_stream_interrupted"


def test_openai_adapter_output_token_budget():
    adapter = OpenAICompatibleAdapter("m", "k", "http://x", max_output_tokens=2)
    fake = _FakeHTTPClient([_FakeResponse(_openai_payload("x" * 100))])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=fake):
        assert len(adapter.generate("q", [], [])) <= 8


def test_anthropic_adapter_generate():
    adapter = AnthropicAdapter("claude-x", "key")
    fake = _FakeHTTPClient([_FakeResponse({"content": [{"type": "text", "text": "hi there"}]})])
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=fake):
        assert adapter.generate("q", [], []) == "hi there"


def test_gemini_adapter_generate():
    adapter = GeminiAdapter("gem-x", "key")
    payload = {"candidates": [{"content": {"parts": [{"text": "gem answer"}]}}]}
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=_FakeHTTPClient([_FakeResponse(payload)])):
        assert adapter.generate("q", [], []) == "gem answer"


def test_ollama_adapter_generate():
    adapter = OllamaAdapter("llama3.1:8b")
    payload = {"message": {"content": "ollama answer"}}
    with mock.patch("rag_assistant.llm.httpx.Client", return_value=_FakeHTTPClient([_FakeResponse(payload)])):
        assert adapter.generate("q", [], []) == "ollama answer"
