"""SQLite persistence for OHLCV + indicators + fundamentals snapshots, plus
the news digest and LLM cascade outputs (sentiment/debate/trader) for every
cycle that reaches them. Storage is free, unlike LLM tokens, so this keeps a
full audit trail and feeds Phase 4's planned similarity retrieval over past
situations. Plain stdlib sqlite3 for Phase 1 — swap for Turso/libSQL later
(Phase 3+) once we need native vector search for the semantic cache; the
schema below is designed to migrate cleanly since it's just relational tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .debate import DebateState, TraderDecision
from .gate import GateResult
from .news_analyst import SentimentSignal

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    macd REAL, macd_signal REAL, macd_hist REAL, rsi REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    trailing_pe REAL, forward_pe REAL, market_cap REAL, beta REAL,
    PRIMARY KEY (ticker, as_of)
);

CREATE TABLE IF NOT EXISTS decisions (
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

CREATE TABLE IF NOT EXISTS debate_turns (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    round INTEGER NOT NULL,
    side TEXT NOT NULL,
    stance TEXT, key_point TEXT, confidence TEXT,
    PRIMARY KEY (ticker, as_of, round, side)
);

-- Judge output over a (ticker, as_of) decision. Keyed on judge_model too, so
-- re-judging with a different/bigger model doesn't clobber a prior judgment.
CREATE TABLE IF NOT EXISTS judgments (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    bull_bear_distinct TEXT, indicator_consistent TEXT,
    news_fidelity TEXT, trader_consistent TEXT,
    overall_score REAL, notes TEXT,
    judged_at TEXT,
    PRIMARY KEY (ticker, as_of, judge_model)
);
"""


def get_connection(db_path: str | Path = "data/edge_analyst.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def save_bars(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """`df` must already have macd/macd_signal/macd_hist/rsi columns merged
    in (see pipeline.py) — this function just persists, it doesn't compute."""
    rows = [
        (
            ticker,
            idx.strftime("%Y-%m-%d"),
            row.open,
            row.high,
            row.low,
            row.close,
            int(row.volume),
            row.macd,
            row.macd_signal,
            row.macd_hist,
            row.rsi,
        )
        for idx, row in df.iterrows()
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO bars
           (ticker, date, open, high, low, close, volume,
            macd, macd_signal, macd_hist, rsi)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def save_fundamentals(
    conn: sqlite3.Connection, ticker: str, as_of: str, f: dict
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO fundamentals
           (ticker, as_of, trailing_pe, forward_pe, market_cap, beta)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            ticker,
            as_of,
            f.get("trailing_pe"),
            f.get("forward_pe"),
            f.get("market_cap"),
            f.get("beta"),
        ),
    )
    conn.commit()


def save_decision(
    conn: sqlite3.Connection,
    ticker: str,
    as_of: str,
    close: float,
    rsi: float,
    macd_hist: float,
    gate_result: GateResult,
    news_text: str | None,
    sentiment: SentimentSignal | None,
    trader: TraderDecision | None,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO decisions
           (ticker, as_of, close, rsi, macd_hist, gate_reasons, news_text,
            sentiment_label, sentiment_score, sentiment_confidence, sentiment_rationale,
            trader_action, trader_reasoning,
            trader_entry_price, trader_stop_loss, trader_position_sizing)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker,
            as_of,
            close,
            rsi,
            macd_hist,
            ",".join(gate_result.reasons),
            news_text,
            sentiment.label if sentiment else None,
            sentiment.score if sentiment else None,
            sentiment.confidence if sentiment else None,
            sentiment.rationale if sentiment else None,
            trader.action if trader else None,
            trader.reasoning if trader else None,
            trader.entry_price if trader else None,
            trader.stop_loss if trader else None,
            trader.position_sizing if trader else None,
        ),
    )
    conn.commit()


def save_debate_turns(
    conn: sqlite3.Connection, ticker: str, as_of: str, history: list[DebateState]
) -> None:
    """One row per side per round — richer than what gets resent to the model
    each round (see debate.py's DebateState overwrite-not-append design);
    storage is free, so we keep the whole debate, not just the final round."""
    rows = []
    for state in history:
        rows.append(
            (
                ticker,
                as_of,
                state.round,
                "bull",
                state.bull.stance,
                state.bull.key_point,
                state.bull.confidence,
            )
        )
        rows.append(
            (
                ticker,
                as_of,
                state.round,
                "bear",
                state.bear.stance,
                state.bear.key_point,
                state.bear.confidence,
            )
        )
    conn.executemany(
        """INSERT OR REPLACE INTO debate_turns
           (ticker, as_of, round, side, stance, key_point, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


_DECISION_COLUMNS = [
    "ticker",
    "as_of",
    "close",
    "rsi",
    "macd_hist",
    "gate_reasons",
    "news_text",
    "sentiment_label",
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_rationale",
    "trader_action",
    "trader_reasoning",
    "trader_entry_price",
    "trader_stop_loss",
    "trader_position_sizing",
]


def fetch_decisions_for_judging(
    conn: sqlite3.Connection, limit: int = 20
) -> list[dict]:
    """One dict per (ticker, as_of) decision, most recent first, with its full
    debate history nested under "debate_turns" — everything eval/rubric.py's
    build_judge_prompt needs, no further DB access required from the judge."""
    rows = conn.execute(
        f"""SELECT {", ".join(_DECISION_COLUMNS)}
            FROM decisions ORDER BY as_of DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    records = [dict(zip(_DECISION_COLUMNS, row, strict=True)) for row in rows]

    turns_by_key: dict[tuple[str, str], list[dict]] = {}
    turn_rows = conn.execute(
        """SELECT ticker, as_of, round, side, stance, key_point, confidence
           FROM debate_turns ORDER BY ticker, as_of, round, side"""
    ).fetchall()
    for ticker, as_of, round_, side, stance, key_point, confidence in turn_rows:
        turns_by_key.setdefault((ticker, as_of), []).append(
            {
                "round": round_,
                "side": side,
                "stance": stance,
                "key_point": key_point,
                "confidence": confidence,
            }
        )

    for record in records:
        record["debate_turns"] = turns_by_key.get(
            (record["ticker"], record["as_of"]), []
        )
    return records


def save_judgment(
    conn: sqlite3.Connection,
    ticker: str,
    as_of: str,
    judge_model: str,
    judgment: dict,
    judged_at: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO judgments
           (ticker, as_of, judge_model, bull_bear_distinct, indicator_consistent,
            news_fidelity, trader_consistent, overall_score, notes, judged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker,
            as_of,
            judge_model,
            judgment.get("bull_bear_distinct"),
            judgment.get("indicator_consistent"),
            judgment.get("news_fidelity"),
            judgment.get("trader_consistent"),
            judgment.get("overall_score"),
            judgment.get("notes"),
            judged_at,
        ),
    )
    conn.commit()
