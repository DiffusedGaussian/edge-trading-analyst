"""Market data fetch wrapper around yfinance. Pure plumbing — swap this
module out later for Alpaca (Phase 5, when we need real paper-broker fills)
without touching anything downstream, since callers only see the DataFrame
shape defined here.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Daily OHLCV bars for one ticker, indexed by date, oldest first."""
    df = yf.Ticker(ticker).history(period=f"{lookback_days}d", interval="1d")
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_fundamentals(ticker: str) -> dict:
    """A handful of snapshot fundamentals. Best-effort — yfinance's fast_info
    doesn't guarantee every field is populated for every ticker."""
    info = yf.Ticker(ticker).get_info()
    return {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "market_cap": info.get("marketCap"),
        "beta": info.get("beta"),
    }
