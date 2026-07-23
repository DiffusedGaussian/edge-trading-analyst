"""Direct HTTP client for llama-server's OpenAI-compatible endpoint. No agent
framework in between — this keeps prompt shape and turn count fully under
our control (see brief Section 5, tech-stack rationale)."""

from __future__ import annotations

import requests


def chat_completion(
    messages: list[dict],
    base_url: str,
    model: str = "local",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
