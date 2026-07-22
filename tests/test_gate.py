"""Smoke tests over the materiality gate.

Pins the two decisions that matter: crossings fire once per *transition* (not per
day the level holds), and gate() ORs independent triggers (any one is enough).
"""

from __future__ import annotations

import pandas as pd

from edge_analyst.gate import (
    GateResult,
    crossed_above,
    crossed_below,
    gate,
    price_move_exceeds,
)


def test_crossed_below_fires_once_on_transition():
    # 35 -> 25 crosses below 30 once; staying under 30 must not re-fire.
    s = pd.Series([35, 25, 24, 23], dtype=float)
    out = crossed_below(s, 30)
    assert list(out) == [False, True, False, False]


def test_crossed_above_fires_once_on_transition():
    s = pd.Series([65, 75, 76, 77], dtype=float)
    out = crossed_above(s, 70)
    assert list(out) == [False, True, False, False]


def test_price_move_exceeds_is_a_level_check():
    # -4% then +4% both exceed a 3% threshold; it's a level check with no
    # persistent state, so consecutive large moves each fire independently.
    close = pd.Series([100, 96, 100], dtype=float)
    out = price_move_exceeds(close, 0.03)
    assert list(out) == [False, True, True]


def test_gate_quiet_when_nothing_triggers():
    close = pd.Series([100, 100.5, 100.7], dtype=float)
    hist = pd.Series([1.0, 1.0, 1.0])  # no zero-cross
    rsi_series = pd.Series([50, 51, 52], dtype=float)  # mid-band
    result = gate(close, hist, rsi_series)
    assert isinstance(result, GateResult)
    assert result.material is False
    assert result.reasons == []


def test_gate_ors_multiple_simultaneous_triggers():
    # Last row: MACD histogram crosses above 0 AND price jumps >3% -> both fire,
    # and material is True because *any* trigger is enough.
    close = pd.Series([100, 100, 110], dtype=float)  # +10% on the last bar
    hist = pd.Series([-1.0, -0.5, 0.5])  # bearish -> bullish cross
    rsi_series = pd.Series([50, 55, 60], dtype=float)  # nothing at the RSI bands
    result = gate(close, hist, rsi_series)
    assert result.material is True
    assert "macd_bullish_crossover" in result.reasons
    assert "price_move" in result.reasons
