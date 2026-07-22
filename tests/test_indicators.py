"""Smoke tests over the deterministic indicators.

These pin the *design decisions* the docstrings call out — not exhaustive numeric
coverage, just enough that CI catches a regression in the load-bearing behavior
(EMA recency-weighting, RSI's RS->inf boundary, MACD crossover sign).
"""

from __future__ import annotations

import pandas as pd

from edge_analyst.indicators import ema, macd, rsi, sma


def test_sma_hard_cutoff_and_nan_warmup():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, window=3)
    # First window-1 rows are NaN (not enough history yet).
    assert out.iloc[:2].isna().all()
    # Equal-weight mean of the trailing window; older prices drop out entirely.
    assert out.iloc[2] == 2.0  # (1+2+3)/3
    assert out.iloc[4] == 4.0  # (3+4+5)/3


def test_ema_weights_recent_more_than_sma():
    # A step up: EMA should sit above the equal-weight SMA on the way up,
    # because it weights the recent higher prices more heavily.
    s = pd.Series([10, 10, 10, 20, 20], dtype=float)
    e = ema(s, span=3)
    a = sma(s, window=3)
    assert e.iloc[3] > a.iloc[3]


def test_rsi_all_gains_saturates_at_100():
    # Monotonically rising series -> avg_loss == 0 -> RS = inf -> RSI resolves to 100
    # with no special-casing (the boundary the docstring promises).
    s = pd.Series(range(1, 30), dtype=float)
    r = rsi(s, period=14)
    assert r.iloc[-1] == 100.0


def test_rsi_all_losses_floors_at_zero():
    s = pd.Series(range(30, 1, -1), dtype=float)
    r = rsi(s, period=14)
    assert r.iloc[-1] == 0.0


def test_macd_histogram_sign_tracks_momentum():
    # Sustained uptrend -> fast EMA outruns slow -> macd line positive.
    s = pd.Series(range(1, 60), dtype=float)
    df = macd(s)
    assert set(df.columns) == {"macd", "signal", "histogram"}
    assert df["macd"].iloc[-1] > 0
    # histogram is exactly macd - signal by construction.
    last = df.iloc[-1]
    assert last["histogram"] == last["macd"] - last["signal"]
