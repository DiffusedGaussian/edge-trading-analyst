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


def analyze_ticker(ticker: str, lookback_days: int) -> TickerSnapshot:
    ohlcv = data_source.fetch_ohlcv(ticker, lookback_days)
    if ohlcv.empty:
        return TickerSnapshot(
            ticker=ticker,
            merged=None,
            close=None,
            rsi=None,
            macd_hist=None,
            gate_result=GateResult(material=False, reasons=[]),
        )

    macd_df = macd(ohlcv["close"])
    rsi_series = rsi(ohlcv["close"])
    merged = ohlcv.assign(
        macd=macd_df["macd"],
        macd_signal=macd_df["signal"],
        macd_hist=macd_df["histogram"],
        rsi=rsi_series,
    )

    gate_result = gate(merged["close"], merged["macd_hist"], merged["rsi"])

    return TickerSnapshot(
        ticker=ticker,
        merged=merged,
        close=float(merged["close"].iloc[-1]),
        rsi=float(merged["rsi"].iloc[-1]),
        macd_hist=float(merged["macd_hist"].iloc[-1]),
        gate_result=gate_result,
    )
