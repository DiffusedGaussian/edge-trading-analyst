"""Deterministic technical indicators. No LLM anywhere in this file."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Equal-weight average of the last `window` closes.

    Hard cutoff: a price drops out completely once it's `window` rows old.
    First `window - 1` rows are NaN (not enough history yet) — fine for us
    since lookback windows are long relative to any window size we use.
    """
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Recency-weighted average: EMA_t = alpha*price_t + (1-alpha)*EMA_(t-1).

    Unlike sma(), no hard cutoff — every past price keeps a nonzero (but
    geometrically shrinking) weight forever. `span` maps to alpha via
    alpha = 2/(span+1), the convention that makes an N-period EMA roughly
    as "reactive" as an N-period SMA. `adjust=False` is required to get
    this exact recursive definition; pandas' default (`adjust=True`)
    bias-corrects the early rows instead, which is a different (if
    converging) formula.
    """
    return series.ewm(span=span, adjust=False).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal_span: int = 9
) -> pd.DataFrame:
    """Momentum vs. trend, via the gap between a fast and a slow EMA.

    `macd` line > 0 means recent price is outrunning the longer trend
    (bullish momentum); < 0 means momentum is fading. The `macd` line is
    itself noisy, so `signal` (a further EMA of it) is a smoothed baseline
    to compare against — `histogram` (macd - signal) flipping sign is the
    crossover, and is the deterministic trigger Phase 2's gate watches for.
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Ratio of up-move strength to down-move strength, scaled to 0-100.

    `gains`/`losses` are zero-filled (not dropped) on the "wrong" days so
    both series stay full-length and aligned with `series` before
    averaging. RS = avg_gain/avg_loss maps monotonically onto 0-100 via
    100 - 100/(1+RS): all-losses -> 0, balanced -> 50, all-gains -> 100.
    avg_loss == 0 makes RS = inf, which resolves cleanly to RSI = 100 —
    no special-casing needed.
    """
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def interpret_rsi(value: float, low: float = 30, high: float = 70) -> str:
    """Deterministic label for an RSI value, using the same 30/70 bands the
    gate fires on. Handed to the LLM alongside the raw number so a small model
    doesn't have to make the numeric judgment it's unreliable at (e.g. calling
    a neutral 55 "bearish")."""
    if value < low:
        return "oversold"
    if value > high:
        return "overbought"
    return "neutral"


def interpret_macd_hist(value: float) -> str:
    """Deterministic label for the MACD histogram. Its sign *is* the meaning:
    positive = fast EMA above signal = bullish momentum, negative = bearish.
    Prevents the small model from mislabeling a positive histogram as bearish."""
    if value > 0:
        return "bullish momentum"
    if value < 0:
        return "bearish momentum"
    return "flat"


def format_market_context(
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
) -> str:
    """The shared indicator block for every agent prompt (analyst, debate,
    trader). Each raw number carries its deterministic interpretation so the
    model only does language synthesis, never the numeric judgment. One home
    for this so the three prompts can't drift apart."""
    reasons = ", ".join(fired_reasons) if fired_reasons else "none"
    return (
        f"Ticker: {ticker}\n"
        f"Current close: ${close:.2f}\n"
        f"RSI: {rsi_value:.1f} ({interpret_rsi(rsi_value)})\n"
        f"MACD histogram: {macd_hist:.3f} ({interpret_macd_hist(macd_hist)})\n"
        f"Triggered rules: {reasons}"
    )
