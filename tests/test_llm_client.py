"""Tests over the llama-server HTTP boundary: what goes out in the request body
and what gets surfaced from the response.

Both matter for eval, not runtime. A missing `seed` makes a run irreproducible;
a discarded `finish_reason` makes a truncated response indistinguishable from a
model that simply can't follow the output format.
"""

from __future__ import annotations

import pytest

from edge_analyst.llm_client import (
    Completion,
    GenSettings,
    chat_completion,
    chat_completion_full,
)

_MESSAGES = [{"role": "user", "content": "hi"}]


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _capture(monkeypatch, payload: dict) -> dict:
    """Patches the HTTP boundary and returns a dict that the sent request body
    lands in, so a test can assert on what was actually transmitted."""
    sent: dict = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return _FakeResponse(payload)

    monkeypatch.setattr("edge_analyst.llm_client.requests.post", fake_post)
    return sent


_MINIMAL_PAYLOAD = {"choices": [{"message": {"content": "ok"}}]}

_FULL_PAYLOAD = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 512},
    "timings": {"prompt_ms": 240.5, "predicted_ms": 8100.0},
}


def test_seed_absent_from_body_when_unset(monkeypatch):
    """A literal null seed is rejected by stricter OpenAI-compatible servers,
    so the key must be missing, not None."""
    sent = _capture(monkeypatch, _MINIMAL_PAYLOAD)
    chat_completion_full(_MESSAGES, "http://x", settings=GenSettings())
    assert "seed" not in sent


def test_seed_present_in_body_when_set(monkeypatch):
    sent = _capture(monkeypatch, _MINIMAL_PAYLOAD)
    chat_completion_full(_MESSAGES, "http://x", settings=GenSettings(seed=1234))
    assert sent["seed"] == 1234


def test_settings_reach_the_body(monkeypatch):
    sent = _capture(monkeypatch, _MINIMAL_PAYLOAD)
    chat_completion_full(
        _MESSAGES, "http://x", settings=GenSettings(temperature=0.0, max_tokens=64)
    )
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 64


def test_defaults_match_the_previous_hardcoded_values(monkeypatch):
    """Runtime behaviour must be unchanged for callers that pass no settings."""
    sent = _capture(monkeypatch, _MINIMAL_PAYLOAD)
    chat_completion_full(_MESSAGES, "http://x")
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 512
    assert "seed" not in sent


def test_response_without_timings_or_usage_parses_with_none_fields(monkeypatch):
    """`timings` is a llama.cpp extension, not standard — a server without it
    must not raise."""
    _capture(monkeypatch, _MINIMAL_PAYLOAD)
    completion = chat_completion_full(_MESSAGES, "http://x")
    assert completion == Completion(
        content="ok",
        finish_reason="unknown",
        prompt_tokens=None,
        completion_tokens=None,
        prompt_ms=None,
        predicted_ms=None,
    )


def test_finish_reason_and_timings_are_surfaced(monkeypatch):
    _capture(monkeypatch, _FULL_PAYLOAD)
    completion = chat_completion_full(_MESSAGES, "http://x")
    assert completion.finish_reason == "length"
    assert completion.prompt_tokens == 120
    assert completion.completion_tokens == 512
    assert completion.prompt_ms == 240.5
    assert completion.predicted_ms == 8100.0


def test_chat_completion_still_returns_just_content(monkeypatch):
    _capture(monkeypatch, _FULL_PAYLOAD)
    assert chat_completion(_MESSAGES, "http://x") == "ok"


def test_raise_for_status_is_honoured(monkeypatch):
    class _Failing(_FakeResponse):
        def raise_for_status(self):
            raise RuntimeError("502")

    monkeypatch.setattr(
        "edge_analyst.llm_client.requests.post",
        lambda url, json, timeout: _Failing({}),
    )
    with pytest.raises(RuntimeError):
        chat_completion_full(_MESSAGES, "http://x")
