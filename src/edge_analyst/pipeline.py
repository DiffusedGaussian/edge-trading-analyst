"""Phase 1 pipeline: data -> indicators -> persist. No LLM. Wires together
data_source.py (infra), indicators.py (core), store.py (infra)."""

from __future__ import annotations

import datetime as dt

from . import data_source, store
from .config import Config, load_config
from .gate import GateResult, gate
from .indicators import macd, rsi


def run_ticker(conn, ticker: str, lookback_days: int) -> tuple[int, GateResult]:
    """Returns (bar-rows persisted, gate outcome) for this ticker."""
    ohlcv = data_source.fetch_ohlcv(ticker, lookback_days)
    if ohlcv.empty:
        return 0, GateResult(material=False, reasons=[])

    macd_df = macd(ohlcv["close"])
    rsi_series = rsi(ohlcv["close"])

    merged = ohlcv.assign(
        macd=macd_df["macd"],
        macd_signal=macd_df["signal"],
        macd_hist=macd_df["histogram"],
        rsi=rsi_series,
    )
    store.save_bars(conn, ticker, merged)

    fundamentals = data_source.fetch_fundamentals(ticker)
    today = dt.date.today().isoformat()
    store.save_fundamentals(conn, ticker, today, fundamentals)

    gate_result = gate(merged["close"], merged["macd_hist"], merged["rsi"])
    return len(merged), gate_result


def run_watchlist(config: Config | None = None) -> dict[str, tuple[int, GateResult]]:
    config = config or load_config()
    conn = store.get_connection()
    results = {}
    for ticker in config.tickers:
        results[ticker] = run_ticker(conn, ticker, config.lookback_days)
    conn.close()
    return results


if __name__ == "__main__":
    for ticker, (n_rows, gate_result) in run_watchlist().items():
        status = f"MATERIAL {gate_result.reasons}" if gate_result.material else "quiet"
        print(f"{ticker}: {n_rows} bars persisted — gate: {status}")
