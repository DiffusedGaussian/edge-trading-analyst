"""The one orchestrator. A cycle is: analyze_ticker (deterministic core) ->
persist bars/fundamentals -> if the gate is material AND news + a model
endpoint are available, run the LLM cascade (analyst -> debate -> trader).
run_watchlist just loops run_cycle over the config tickers.

The deterministic path (up to and including the gate) needs no LLM and runs
for the whole watchlist for free. The LLM cascade only runs on a material
gate, which is the entire point of Phase 2.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import data_source, store
from .analysis import TickerSnapshot, analyze_ticker
from .analyst import build_sentiment_prompt, parse_sentiment_response
from .config import Config, load_config
from .debate import (
    DebateState,
    TraderDecision,
    build_trader_prompt,
    parse_trader_response,
    run_debate,
)
from .llm_client import chat_completion


@dataclass
class CycleResult:
    snapshot: TickerSnapshot
    # The cascade fields stay None when the gate is quiet, or when the gate
    # is material but no news / model endpoint was supplied (batch runs).
    sentiment: object | None = None
    debate: DebateState | None = None
    trader: TraderDecision | None = None


def run_cycle(
    conn,
    ticker: str,
    lookback_days: int,
    news_text: str | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> CycleResult:
    """`force=True` runs the LLM cascade even on a quiet gate — for
    testing/demoing the full chain on a day nothing triggers. The real gate
    result is still recorded on the snapshot; force only bypasses the early exit."""
    snapshot = analyze_ticker(ticker, lookback_days)
    if not snapshot.has_data:
        return CycleResult(snapshot=snapshot)

    store.save_bars(conn, ticker, snapshot.merged)
    fundamentals = data_source.fetch_fundamentals(ticker)
    store.save_fundamentals(conn, ticker, dt.date.today().isoformat(), fundamentals)

    # Gate is the cost lever: no material change -> exit before any LLM call.
    if not snapshot.gate_result.material and not force:
        return CycleResult(snapshot=snapshot)

    # Material change. Reasoning needs a model endpoint; batch runs pass none,
    # so they stop at the (persisted) deterministic result. Given an endpoint,
    # fetch news automatically unless the caller supplied it (e.g. for testing).
    if base_url is None:
        return CycleResult(snapshot=snapshot)
    if news_text is None:
        news_text = data_source.fetch_news(ticker)

    reasons = snapshot.gate_result.reasons
    analyst_messages = build_sentiment_prompt(
        ticker, snapshot.close, snapshot.rsi, snapshot.macd_hist, reasons, news_text
    )
    sentiment = parse_sentiment_response(
        chat_completion(analyst_messages, base_url=base_url)
    )

    # NOTE: how the debate should consume large/multi-article news vs. the
    # analyst's distilled summary is an open design decision (not resolved).
    # For now the debate keeps taking the raw news_text, as originally built.
    debate = run_debate(
        ticker,
        snapshot.close,
        snapshot.rsi,
        snapshot.macd_hist,
        reasons,
        news_text,
        base_url,
    )

    trader_messages = build_trader_prompt(
        ticker, snapshot.close, snapshot.rsi, snapshot.macd_hist, reasons, debate
    )
    trader = parse_trader_response(chat_completion(trader_messages, base_url=base_url))

    return CycleResult(
        snapshot=snapshot, sentiment=sentiment, debate=debate, trader=trader
    )


def run_watchlist(config: Config | None = None) -> dict[str, CycleResult]:
    """Batch, deterministic-only (no news source yet) — every ticker runs
    through the gate and persists, none reach the LLM cascade."""
    config = config or load_config()
    conn = store.get_connection()
    results = {}
    for ticker in config.tickers:
        results[ticker] = run_cycle(conn, ticker, config.lookback_days)
    conn.close()
    return results


def _print_cycle(ticker: str, result: CycleResult) -> None:
    snap = result.snapshot
    if not snap.has_data:
        print(f"{ticker}: no data returned")
        return
    gate_status = (
        f"MATERIAL {snap.gate_result.reasons}" if snap.gate_result.material else "quiet"
    )
    print(
        f"{ticker}: close=${snap.close:.2f} RSI={snap.rsi:.1f} "
        f"MACD_hist={snap.macd_hist:.3f} — gate: {gate_status}"
    )
    if result.sentiment is not None:
        print(f"  analyst: {result.sentiment}")
    if result.debate is not None:
        print(f"  debate (round {result.debate.round}):")
        print(f"    bull: {result.debate.bull}")
        print(f"    bear: {result.debate.bear}")
    if result.trader is not None:
        print(f"  trader: {result.trader}")


if __name__ == "__main__":
    import sys

    # No args -> batch watchlist (deterministic only, no news source).
    # TICKER [base_url] [--force] -> single ticker, full live cascade with
    #   auto-fetched news. --force runs the cascade even on a quiet gate.
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    if args:
        ticker = args[0]
        base_url = args[1] if len(args) > 1 else "http://localhost:8080"
        config = load_config()
        conn = store.get_connection()
        result = run_cycle(
            conn, ticker, config.lookback_days, base_url=base_url, force=force
        )
        conn.close()
        _print_cycle(ticker, result)
    else:
        for ticker, result in run_watchlist().items():
            _print_cycle(ticker, result)
