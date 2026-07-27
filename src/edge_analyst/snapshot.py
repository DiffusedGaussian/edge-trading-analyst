"""Shared deterministic core: fetch -> indicators -> gate, as one function.

Previously this exact sequence was reimplemented inline in three places
(pipeline batch run, news_analyst __main__, cycle draft). It lives here once now;
everything that needs "the current deterministic picture for a ticker" calls
analyze_ticker() and gets a TickerSnapshot. No LLM, no persistence — just the
computation. Callers decide what to persist and whether to spend LLM tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import data_source
from .gate import GateResult, gate
from .indicators import macd, rsi


@dataclass
class TickerSnapshot:
    ticker: str
    merged: pd.DataFrame | None  # full history + indicator columns; None if no data
    close: float | None
    rsi: float | None
    macd_hist: float | None
    gate_result: GateResult

    @property
    def has_data(self) -> bool:
        return self.merged is not None


def _no_data(ticker: str) -> TickerSnapshot:
    return TickerSnapshot(
        ticker=ticker,
        merged=None,
        close=None,
        rsi=None,
        macd_hist=None,
        gate_result=GateResult(material=False, reasons=[]),
    )


def analyze_ticker(ticker: str, lookback_days: int) -> TickerSnapshot:
    ohlcv = data_source.fetch_ohlcv(ticker, lookback_days)

    # yfinance can hand back a still-forming "today" bar with no close yet
    # while the market is open -- interval="1d" doesn't guarantee the last
    # row is a *finished* day. A NaN close is a valid Python float, not None,
    # so nothing upstream would notice: it silently becomes SQL NULL on
    # persistence (sqlite3 launders NaN -> NULL on write) and renders as the
    # literal string "$nan" in format_market_context, handed to the model as
    # ground truth. Trim trailing incomplete rows so every reported "close" is
    # a finished bar's real price.
    while len(ohlcv) and pd.isna(ohlcv["close"].iloc[-1]):
        ohlcv = ohlcv.iloc[:-1]
    if ohlcv.empty:
        return _no_data(ticker)

    macd_df = macd(ohlcv["close"])
    rsi_series = rsi(ohlcv["close"])
    merged = ohlcv.assign(
        macd=macd_df["macd"],
        macd_signal=macd_df["signal"],
        macd_hist=macd_df["histogram"],
        rsi=rsi_series,
    )

    close = float(merged["close"].iloc[-1])
    rsi_value = float(merged["rsi"].iloc[-1])
    macd_hist_value = float(merged["macd_hist"].iloc[-1])
    # Defense in depth beyond the trailing-close trim above: e.g. too little
    # warm-up history could leave rsi/macd_hist NaN even with a real close.
    # Any of the three reaching here as NaN must be treated as no data, not
    # silently persisted/prompted as if it were a real reading.
    if pd.isna(close) or pd.isna(rsi_value) or pd.isna(macd_hist_value):
        return _no_data(ticker)

    gate_result = gate(merged["close"], merged["macd_hist"], merged["rsi"])

    return TickerSnapshot(
        ticker=ticker,
        merged=merged,
        close=close,
        rsi=rsi_value,
        macd_hist=macd_hist_value,
        gate_result=gate_result,
    )
