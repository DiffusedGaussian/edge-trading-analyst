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


def fetch_news(ticker: str, max_items: int = 5) -> str:
    """Recent news as a bounded, pre-summarized digest — title + yfinance's own
    summary for the top `max_items` items, no full-article crawl. Bounding the
    item count (and using the short `summary`, not the HTML `description`) keeps
    the token cost of feeding this into the LLM predictable. yfinance news is
    only loosely ticker-specific, so treat it as ambient context, not gospel.
    Returns a sentinel string when nothing is available."""
    items = yf.Ticker(ticker).news or []
    lines = []
    for item in items[:max_items]:
        content = item.get("content", {})
        title = (content.get("title") or "").strip()
        summary = (content.get("summary") or "").strip()
        if not title:
            continue
        lines.append(f"- {title}: {summary}" if summary else f"- {title}")
    return "\n".join(lines) if lines else "No recent news available."
