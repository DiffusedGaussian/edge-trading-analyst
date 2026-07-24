"""Round-trip tests for the decisions/debate_turns persistence added alongside
the news + LLM-output storage — bars/fundamentals were already exercised
indirectly via pipeline.py before this.
"""

from __future__ import annotations

from edge_analyst import store
from edge_analyst.debate import DebateState, DebateTurn, TraderDecision
from edge_analyst.gate import GateResult
from edge_analyst.news_analyst import SentimentSignal


def test_save_decision_round_trips_all_fields():
    conn = store.get_connection(":memory:")
    gate_result = GateResult(material=True, reasons=["price_move", "rsi_oversold"])
    sentiment = SentimentSignal(
        label="bullish", score=7.0, confidence="high", rationale="a fact"
    )
    trader = TraderDecision(
        action="buy",
        reasoning="strong case",
        entry_price=150.0,
        stop_loss=140.0,
        position_sizing=5.0,
    )

    store.save_decision(
        conn, "AAPL", "2026-07-24T10:00:00", gate_result, "some news", sentiment, trader
    )

    row = conn.execute(
        "SELECT ticker, as_of, gate_reasons, news_text, sentiment_label, "
        "sentiment_score, trader_action, trader_entry_price FROM decisions"
    ).fetchone()
    assert row == (
        "AAPL",
        "2026-07-24T10:00:00",
        "price_move,rsi_oversold",
        "some news",
        "bullish",
        7.0,
        "buy",
        150.0,
    )


def test_save_decision_handles_none_sentiment_and_trader():
    conn = store.get_connection(":memory:")
    gate_result = GateResult(material=False, reasons=[])

    store.save_decision(
        conn, "AAPL", "2026-07-24T10:00:00", gate_result, None, None, None
    )

    row = conn.execute(
        "SELECT sentiment_label, trader_action FROM decisions"
    ).fetchone()
    assert row == (None, None)


def test_save_debate_turns_writes_one_row_per_side_per_round():
    conn = store.get_connection(":memory:")
    history = [
        DebateState(
            round=1,
            bull=DebateTurn(stance="buy", key_point="a", confidence="high"),
            bear=DebateTurn(stance="sell", key_point="b", confidence="high"),
        ),
        DebateState(
            round=2,
            bull=DebateTurn(stance="buy", key_point="c", confidence="high"),
            bear=DebateTurn(stance="hold", key_point="d", confidence="low"),
        ),
    ]

    store.save_debate_turns(conn, "AAPL", "2026-07-24T10:00:00", history)

    rows = conn.execute(
        "SELECT round, side, stance, key_point FROM debate_turns ORDER BY round, side"
    ).fetchall()
    assert rows == [
        (1, "bear", "sell", "b"),
        (1, "bull", "buy", "a"),
        (2, "bear", "hold", "d"),
        (2, "bull", "buy", "c"),
    ]
