"""Round-trip tests for the decisions/debate_turns persistence added alongside
the news + LLM-output storage — bars/fundamentals were already exercised
indirectly via pipeline.py before this.
"""

from __future__ import annotations

import sqlite3

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


# --- model attribution + migration ------------------------------------------

# The decisions/debate_turns DDL as it stood before the `model` column existed.
# Any DB created by an earlier release looks like this, including the live
# baseline — _migrate has to reach it, since CREATE TABLE IF NOT EXISTS won't.
_PRE_MODEL_SCHEMA = """
CREATE TABLE decisions (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    close REAL, rsi REAL, macd_hist REAL,
    gate_reasons TEXT,
    news_text TEXT,
    sentiment_label TEXT, sentiment_score REAL,
    sentiment_confidence TEXT, sentiment_rationale TEXT,
    trader_action TEXT, trader_reasoning TEXT,
    trader_entry_price REAL, trader_stop_loss REAL, trader_position_sizing REAL,
    PRIMARY KEY (ticker, as_of)
);
CREATE TABLE debate_turns (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    round INTEGER NOT NULL,
    side TEXT NOT NULL,
    stance TEXT, key_point TEXT, confidence TEXT,
    PRIMARY KEY (ticker, as_of, round, side)
);
"""


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_migrate_adds_model_column_to_a_preexisting_db(tmp_path):
    db_path = tmp_path / "old.db"
    old = sqlite3.connect(db_path)
    old.executescript(_PRE_MODEL_SCHEMA)
    old.execute(
        "INSERT INTO decisions (ticker, as_of, close) VALUES ('AAPL', 'then', 1.0)"
    )
    old.commit()
    old.close()

    conn = store.get_connection(db_path)

    assert _columns(conn, "decisions").count("model") == 1
    assert _columns(conn, "debate_turns").count("model") == 1
    # Additive only: the pre-existing row survives, with a NULL model.
    assert conn.execute("SELECT close, model FROM decisions").fetchone() == (1.0, None)


def test_migrate_is_idempotent_across_connections(tmp_path):
    db_path = tmp_path / "old.db"
    old = sqlite3.connect(db_path)
    old.executescript(_PRE_MODEL_SCHEMA)
    old.close()

    store.get_connection(db_path).close()
    conn = store.get_connection(db_path)  # second open must not re-ALTER

    assert _columns(conn, "decisions").count("model") == 1


def _save(conn, ticker, as_of, model, action="buy"):
    return store.save_decision(
        conn,
        ticker,
        as_of,
        150.0,
        55.0,
        0.5,
        GateResult(material=True, reasons=["price_move"]),
        "some news",
        SentimentSignal(
            label="bullish", score=7.0, confidence="high", rationale="a fact"
        ),
        TraderDecision(
            action=action,
            reasoning="strong case",
            entry_price=150.0,
            stop_loss=140.0,
            position_sizing=5.0,
        ),
        model,
    )


def test_save_decision_records_the_model():
    conn = store.get_connection(":memory:")
    _save(conn, "AAPL", "2026-07-24T10:00:00", "olmoe-1b-7b")
    assert conn.execute("SELECT model FROM decisions").fetchone()[0] == "olmoe-1b-7b"


def test_save_decision_suffixes_as_of_rather_than_clobbering_another_model():
    conn = store.get_connection(":memory:")
    as_of = "2026-07-24T10:00:00"
    first = _save(conn, "AAPL", as_of, "model-a")
    second = _save(conn, "AAPL", as_of, "model-b")

    assert first == as_of
    assert second == f"{as_of}#2"
    rows = conn.execute("SELECT as_of, model FROM decisions ORDER BY as_of").fetchall()
    assert rows == [(as_of, "model-a"), (f"{as_of}#2", "model-b")]


def test_save_decision_same_model_still_overwrites_in_place():
    conn = store.get_connection(":memory:")
    as_of = "2026-07-24T10:00:00"
    _save(conn, "AAPL", as_of, "model-a", action="buy")
    again = _save(conn, "AAPL", as_of, "model-a", action="sell")

    assert again == as_of
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute("SELECT trader_action FROM decisions").fetchone()[0] == "sell"


def test_save_decision_suffix_keeps_climbing_for_a_third_model():
    conn = store.get_connection(":memory:")
    as_of = "2026-07-24T10:00:00"
    _save(conn, "AAPL", as_of, "model-a")
    _save(conn, "AAPL", as_of, "model-b")
    third = _save(conn, "AAPL", as_of, "model-c")

    assert third == f"{as_of}#3"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 3


def test_save_debate_turns_records_the_model():
    conn = store.get_connection(":memory:")
    history = [
        DebateState(
            round=1,
            bull=DebateTurn(stance="buy", key_point="a", confidence="high"),
            bear=DebateTurn(stance="sell", key_point="b", confidence="high"),
        )
    ]
    store.save_debate_turns(conn, "AAPL", "2026-07-24T10:00:00", history, "olmoe-1b-7b")

    models = conn.execute("SELECT DISTINCT model FROM debate_turns").fetchall()
    assert models == [("olmoe-1b-7b",)]


def test_fetch_decisions_for_judging_filters_on_model():
    conn = store.get_connection(":memory:")
    _save(conn, "AAPL", "2026-07-24T10:00:00", "model-a")
    _save(conn, "MSFT", "2026-07-24T11:00:00", "model-b")

    records = store.fetch_decisions_for_judging(conn, limit=10, model="model-b")

    assert [r["ticker"] for r in records] == ["MSFT"]
    assert records[0]["model"] == "model-b"
    # No filter -> both, unchanged behaviour.
    assert len(store.fetch_decisions_for_judging(conn, limit=10)) == 2


def test_fetch_decisions_for_pairwise_matches_on_ticker_and_day():
    conn = store.get_connection(":memory:")
    # Same ticker, same day, different seconds -> a pair.
    _save(conn, "AAPL", "2026-07-24T10:00:00", "model-a")
    _save(conn, "AAPL", "2026-07-24T14:30:00", "model-b")
    # Same ticker, different days -> not comparable, no pair.
    _save(conn, "MSFT", "2026-07-24T10:00:00", "model-a")
    _save(conn, "MSFT", "2026-07-25T10:00:00", "model-b")
    # Only one model ran this ticker -> no pair.
    _save(conn, "NVDA", "2026-07-24T10:00:00", "model-a")

    pairs = store.fetch_decisions_for_pairwise(conn, "model-a", "model-b")

    assert len(pairs) == 1
    a, b = pairs[0]
    assert a["ticker"] == b["ticker"] == "AAPL"
    assert a["model"] == "model-a"
    assert b["model"] == "model-b"


def test_fetch_decisions_for_pairwise_takes_the_latest_run_per_ticker_day():
    conn = store.get_connection(":memory:")
    _save(conn, "AAPL", "2026-07-24T09:00:00", "model-a", action="buy")
    _save(conn, "AAPL", "2026-07-24T15:00:00", "model-a", action="sell")
    _save(conn, "AAPL", "2026-07-24T16:00:00", "model-b")

    pairs = store.fetch_decisions_for_pairwise(conn, "model-a", "model-b")

    assert len(pairs) == 1
    assert pairs[0][0]["as_of"] == "2026-07-24T15:00:00"
