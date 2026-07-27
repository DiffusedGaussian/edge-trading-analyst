"""Tests over analyze_ticker — the shared deterministic core (fetch -> indicators
-> gate). No existing test file exercised this directly before; coverage lived
only indirectly through test_pipeline.py's mocked run_cycle calls.

The NaN-trailing-close case pins a live bug: yfinance can hand back a
still-forming "today" bar with no close yet while the market is open (interval
"1d" doesn't guarantee the last row is a *finished* day). A NaN close is a
valid Python float, not None, so nothing upstream would notice — it silently
becomes SQL NULL on persistence (sqlite3 launders NaN -> NULL on write) and
renders as the literal string "$nan" in format_market_context, handed to the
model as ground truth. Reproduced live: every intraday decision on the Jetson
had close=NULL while rsi/macd_hist were real, because pandas' ewm() doesn't
propagate one trailing NaN input backward through the whole smoothed series.
"""

from __future__ import annotations

import math

import pandas as pd

from edge_analyst import data_source
from edge_analyst.snapshot import analyze_ticker

_FUNDAMENTALS_IRRELEVANT = {}  # analyze_ticker never touches fundamentals


def _ohlcv(n_days: int = 60, trailing_nan_days: int = 0) -> pd.DataFrame:
    """A gentle uptrend, long enough to warm up RSI/MACD, with the last
    `trailing_nan_days` closes (and only closes -- open/high/low/volume stay
    real, matching what yfinance actually returns for a still-forming bar)
    replaced with NaN."""
    closes = [100.0 + 0.3 * i for i in range(n_days)]
    index = pd.date_range("2026-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "open": [c - 0.05 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n_days,
        },
        index=index,
    )
    for i in range(1, trailing_nan_days + 1):
        df.iloc[-i, df.columns.get_loc("close")] = float("nan")
    return df


def test_analyze_ticker_never_returns_a_nan_close(monkeypatch):
    """The exact live failure: a NaN close must never surface as a Python
    float NaN -- it silently becomes SQL NULL on save and "$nan" in a prompt,
    neither of which raises, so nothing catches it downstream."""
    monkeypatch.setattr(
        data_source, "fetch_ohlcv", lambda *a, **k: _ohlcv(trailing_nan_days=1)
    )

    snapshot = analyze_ticker("NVDA", 60)

    assert snapshot.has_data
    assert snapshot.close is not None
    assert not math.isnan(snapshot.close)
    assert not math.isnan(snapshot.rsi)
    assert not math.isnan(snapshot.macd_hist)


def test_analyze_ticker_uses_the_last_complete_bar_after_trimming(monkeypatch):
    """The reported close is the last *finished* day's price, not a fabricated
    one -- trimming falls back to real history, it doesn't invent a value."""
    complete = _ohlcv(trailing_nan_days=0)
    with_incomplete_bar = _ohlcv(trailing_nan_days=1)
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: with_incomplete_bar)

    snapshot = analyze_ticker("NVDA", 60)

    # The day *before* the NaN one -- both series share this value, since
    # trailing_nan_days=1 only overwrites the final day's close.
    assert snapshot.close == complete["close"].iloc[-2]


def test_analyze_ticker_trims_more_than_one_trailing_nan_row(monkeypatch):
    monkeypatch.setattr(
        data_source, "fetch_ohlcv", lambda *a, **k: _ohlcv(trailing_nan_days=3)
    )

    snapshot = analyze_ticker("NVDA", 60)

    assert snapshot.has_data
    assert not math.isnan(snapshot.close)


def test_analyze_ticker_reports_no_data_when_every_row_is_nan(monkeypatch):
    all_nan = _ohlcv(n_days=5, trailing_nan_days=5)
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: all_nan)

    snapshot = analyze_ticker("NVDA", 5)

    assert not snapshot.has_data
    assert snapshot.close is None
    assert snapshot.rsi is None
    assert snapshot.macd_hist is None


def test_analyze_ticker_with_no_nan_is_unaffected(monkeypatch):
    """The common case -- no trailing NaN at all -- must behave exactly as
    before: this fix must not change output when there is nothing to trim."""
    clean = _ohlcv(trailing_nan_days=0)
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: clean)

    snapshot = analyze_ticker("NVDA", 60)

    assert snapshot.has_data
    assert snapshot.close == clean["close"].iloc[-1]


def test_analyze_ticker_empty_ohlcv_reports_no_data(monkeypatch):
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: pd.DataFrame())

    snapshot = analyze_ticker("NVDA", 60)

    assert not snapshot.has_data
    assert snapshot.close is None
