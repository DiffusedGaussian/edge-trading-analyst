"""Phase 2 materiality gate. Deterministic only — no LLM. Decides whether a
cycle's indicators justify spending an LLM call, or exit for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def crossed_below(series: pd.Series, threshold: float) -> pd.Series:
    """True on the single row where `series` crossed from >=threshold
    yesterday to <threshold today. A transition, not a sustained level —
    firing once per event, not once per day the level holds."""
    return (series.shift(1) >= threshold) & (series < threshold)


def crossed_above(series: pd.Series, threshold: float) -> pd.Series:
    """Mirror of crossed_below: from <=threshold yesterday to >threshold today."""
    return (series.shift(1) <= threshold) & (series > threshold)


def price_move_exceeds(close: pd.Series, pct_threshold: float) -> pd.Series:
    """True on any day the close moved more than `pct_threshold` from the
    prior close. A level check, not a crossing check — unlike RSI/MACD,
    day-over-day % change has no persistent state to falsely re-trigger on."""
    return close.pct_change().abs() > pct_threshold


@dataclass
class GateResult:
    material: bool
    reasons: list[str]


def gate(
    close: pd.Series,
    macd_hist: pd.Series,
    rsi_series: pd.Series,
    rsi_low: float = 30,
    rsi_high: float = 70,
    price_pct: float = 0.03,
) -> GateResult:
    """OR across independent triggers — any one firing is reason enough to
    spend an LLM call; requiring all to agree would miss fast-moving events
    the slower, smoothed indicators haven't caught up to yet."""
    reasons = []
    if crossed_above(macd_hist, 0).iloc[-1]:
        reasons.append("macd_bullish_crossover")
    if crossed_below(macd_hist, 0).iloc[-1]:
        reasons.append("macd_bearish_crossover")
    if crossed_below(rsi_series, rsi_low).iloc[-1]:
        reasons.append("rsi_oversold")
    if crossed_above(rsi_series, rsi_high).iloc[-1]:
        reasons.append("rsi_overbought")
    if price_move_exceeds(close, price_pct).iloc[-1]:
        reasons.append("price_move")
    return GateResult(material=bool(reasons), reasons=reasons)
