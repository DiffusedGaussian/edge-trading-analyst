"""Phase 3 quick-tier news/sentiment analyst. Builds a prompt anchored on
already-computed deterministic indicators (never asks the model to recompute
or recall financial math), and parses its reply forgivingly — sentinel
key-value lines, not JSON, per TradingAgents' proven pattern for small models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm_parsing import extract_field

_SYSTEM_PROMPT = """You are a market sentiment analyst. You are given real, \
already-computed technical indicators for a stock and a news item. Treat the \
indicator values as ground truth — do not recompute or second-guess them. \
Your job is to judge how the news item should be weighed against them, and \
respond in EXACTLY this format, nothing else before or after it:

LABEL: <bullish, bearish, or neutral>
SCORE: <0-10>
CONFIDENCE: <low, medium, or high>
RATIONALE: <one sentence, must reference a specific fact from the input>

Example:
LABEL: bullish
SCORE: 7
CONFIDENCE: high
RATIONALE: RSI crossed above 70 alongside a positive earnings headline."""


def build_sentiment_prompt(
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
    news_text: str,
) -> list[dict]:
    user_prompt = f"""Ticker: {ticker}
Current close: ${close:.2f}
RSI: {rsi_value:.1f}
MACD histogram: {macd_hist:.3f}
Triggered rules: {", ".join(fired_reasons) if fired_reasons else "none"}
News: {news_text}"""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class SentimentSignal:
    label: str
    score: float
    confidence: str
    rationale: str


_LABELS = {"bullish", "bearish", "neutral"}
_CONFIDENCES = {"low", "medium", "high"}


def parse_sentiment_response(text: str) -> SentimentSignal:
    """Forgiving line-scan parser: finds each labeled field anywhere in the
    text (tolerant of preambles/markdown fences around it), and falls back
    to a safe default per-field rather than raising on any single miss."""
    label = (extract_field(text, "LABEL") or "").lower()
    if label not in _LABELS:
        label = "neutral"

    confidence = (extract_field(text, "CONFIDENCE") or "").lower()
    if confidence not in _CONFIDENCES:
        confidence = "low"

    score_raw = extract_field(text, "SCORE")
    try:
        score = max(0.0, min(10.0, float(score_raw)))
    except (TypeError, ValueError):
        score = 5.0

    rationale = extract_field(text, "RATIONALE") or "no rationale parsed"

    return SentimentSignal(
        label=label, score=score, confidence=confidence, rationale=rationale
    )
