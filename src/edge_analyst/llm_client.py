"""Direct HTTP client for llama-server's OpenAI-compatible endpoint. No agent
framework in between — this keeps prompt shape and turn count fully under
our control (see brief Section 5, tech-stack rationale)."""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class GenSettings:
    """Generation knobs, passed as one object so adding another later doesn't
    mean touching every call site in the cascade.

    `seed` defaults to None (the previous behaviour: whatever the server picks),
    but eval runs must set it — without a seed no run is reproducible, and
    "model A beat model B" can't be told apart from sampling noise.
    """

    temperature: float = 0.2
    max_tokens: int = 512
    seed: int | None = None


@dataclass(frozen=True)
class Completion:
    content: str
    # "stop" = the model finished; "length" = it hit max_tokens mid-sentence.
    # The forgiving parsers absorb a truncated response as a pile of fallbacks
    # with no hint of the cause — which is exactly what a long thinking trace
    # from e.g. Qwen3 would produce — so eval needs the reason itself.
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    # llama.cpp's `timings` extension to the OpenAI response shape. Absent on
    # other OpenAI-compatible servers, hence optional. tok/s is derived in the
    # reporting layer, not here: this stays a transport.
    prompt_ms: float | None
    predicted_ms: float | None


def chat_completion_full(
    messages: list[dict],
    base_url: str,
    model: str = "local",
    settings: GenSettings | None = None,
) -> Completion:
    settings = settings or GenSettings()
    body = {
        "model": model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    # Only send `seed` when set: llama.cpp accepts it, but a literal null is
    # rejected by stricter OpenAI-compatible servers.
    if settings.seed is not None:
        body["seed"] = settings.seed

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()

    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    timings = payload.get("timings") or {}
    return Completion(
        content=choice["message"]["content"],
        finish_reason=choice.get("finish_reason") or "unknown",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        prompt_ms=timings.get("prompt_ms"),
        predicted_ms=timings.get("predicted_ms"),
    )


def chat_completion(
    messages: list[dict],
    base_url: str,
    model: str = "local",
    settings: GenSettings | None = None,
) -> str:
    """Back-compat thin wrapper: content only, which is all the runtime cascade
    needs. Eval calls chat_completion_full for the metadata discarded here."""
    return chat_completion_full(messages, base_url, model, settings).content
