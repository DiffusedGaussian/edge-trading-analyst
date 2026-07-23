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


if __name__ == "__main__":
    import sys

    from . import data_source
    from .gate import gate
    from .indicators import macd, rsi
    from .llm_client import chat_completion

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    news_text = sys.argv[2] if len(sys.argv) > 2 else "No major news today."
    base_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8080"

    ohlcv = data_source.fetch_ohlcv(ticker, lookback_days=90)
    macd_df = macd(ohlcv["close"])
    rsi_series = rsi(ohlcv["close"])
    gate_result = gate(ohlcv["close"], macd_df["histogram"], rsi_series)

    messages = build_sentiment_prompt(
        ticker=ticker,
        close=ohlcv["close"].iloc[-1],
        rsi_value=rsi_series.iloc[-1],
        macd_hist=macd_df["histogram"].iloc[-1],
        fired_reasons=gate_result.reasons,
        news_text=news_text,
    )

    print("=== PROMPT ===")
    for m in messages:
        print(f"--- {m['role']} ---\n{m['content']}\n")

    raw = chat_completion(messages, base_url=base_url)
    print("=== RAW MODEL OUTPUT ===")
    print(raw)
    print("\n=== PARSED ===")
    print(parse_sentiment_response(raw))
