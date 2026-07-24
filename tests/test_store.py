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
        conn,
        "AAPL",
        "2026-07-24T10:00:00",
        150.0,
        65.0,
        0.5,
        gate_result,
        "some news",
        sentiment,
        trader,
    )

    row = conn.execute(
        "SELECT ticker, as_of, close, rsi, macd_hist, gate_reasons, news_text, "
        "sentiment_label, sentiment_score, trader_action, trader_entry_price "
        "FROM decisions"
    ).fetchone()
    assert row == (
        "AAPL",
        "2026-07-24T10:00:00",
        150.0,
        65.0,
        0.5,
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
        conn,
        "AAPL",
        "2026-07-24T10:00:00",
        150.0,
        65.0,
        0.5,
        gate_result,
        None,
        None,
        None,
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


def test_fetch_decisions_for_judging_nests_debate_turns():
    conn = store.get_connection(":memory:")
    gate_result = GateResult(material=True, reasons=["price_move"])
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
        conn,
        "AAPL",
        "2026-07-24T10:00:00",
        150.0,
        55.0,
        0.5,
        gate_result,
        "some news",
        sentiment,
        trader,
    )
    store.save_debate_turns(
        conn,
        "AAPL",
        "2026-07-24T10:00:00",
        [
            DebateState(
                round=1,
                bull=DebateTurn(stance="buy", key_point="a", confidence="high"),
                bear=DebateTurn(stance="sell", key_point="b", confidence="high"),
            )
        ],
    )

    records = store.fetch_decisions_for_judging(conn, limit=10)

    assert len(records) == 1
    record = records[0]
    assert record["ticker"] == "AAPL"
    assert record["close"] == 150.0
    assert record["trader_action"] == "buy"
    assert record["debate_turns"] == [
        {
            "round": 1,
            "side": "bear",
            "stance": "sell",
            "key_point": "b",
            "confidence": "high",
        },
        {
            "round": 1,
            "side": "bull",
            "stance": "buy",
            "key_point": "a",
            "confidence": "high",
        },
    ]


def test_fetch_decisions_for_judging_respects_limit_and_recency():
    conn = store.get_connection(":memory:")
    gate_result = GateResult(material=False, reasons=[])
    for i in range(3):
        store.save_decision(
            conn,
            "AAPL",
            f"2026-07-2{i}T10:00:00",
            150.0,
            55.0,
            0.5,
            gate_result,
            None,
            None,
            None,
        )

    records = store.fetch_decisions_for_judging(conn, limit=2)

    assert len(records) == 2
    assert records[0]["as_of"] == "2026-07-22T10:00:00"
    assert records[1]["as_of"] == "2026-07-21T10:00:00"


def test_save_judgment_round_trips():
    conn = store.get_connection(":memory:")
    judgment = {
        "bull_bear_distinct": "no",
        "indicator_consistent": "no",
        "news_fidelity": "yes",
        "trader_consistent": "unknown",
        "overall_score": 3.0,
        "notes": "bull and bear echoed the same key point",
    }

    store.save_judgment(
        conn,
        "AAPL",
        "2026-07-24T10:00:00",
        "qwen2.5-32b-instruct",
        judgment,
        "2026-07-24T11:00:00",
    )

    row = conn.execute(
        "SELECT ticker, as_of, judge_model, bull_bear_distinct, "
        "overall_score, notes FROM judgments"
    ).fetchone()
    assert row == (
        "AAPL",
        "2026-07-24T10:00:00",
        "qwen2.5-32b-instruct",
        "no",
        3.0,
        "bull and bear echoed the same key point",
    )
