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

-- `model` records which llama-server model produced the LLM outputs on this
-- row. Without it a decision is unattributable and no model-vs-model
-- comparison is possible. Not part of the PK — see save_decision's collision
-- guard for why, and _migrate for how it reaches pre-existing DBs.
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
    model TEXT,
    PRIMARY KEY (ticker, as_of)
);

CREATE TABLE IF NOT EXISTS debate_turns (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    round INTEGER NOT NULL,
    side TEXT NOT NULL,
    stance TEXT, key_point TEXT, confidence TEXT,
    model TEXT,
    PRIMARY KEY (ticker, as_of, round, side)
);

-- Tier 0 fixture runs. Synthetic fixtures deliberately never touch `decisions`:
-- that table is real market history, and a TESTA row in it would corrupt every
-- replayed-real comparison downstream. These two tables are the synthetic side
-- of that boundary.
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,        -- iso timestamp + model_name slug
    model_name TEXT NOT NULL,
    base_url TEXT, stage TEXT,
    k INTEGER, seed INTEGER, temperature REAL,
    started_at TEXT, finished_at TEXT,
    git_sha TEXT                    -- so a scorecard is traceable to code
);

CREATE TABLE IF NOT EXISTS eval_samples (
    run_id TEXT NOT NULL,
    fixture_id TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    stage TEXT NOT NULL,
    raw_output TEXT,                -- the full raw text; becomes a CI fixture
    parsed_json TEXT,               -- parsed dataclass as JSON
    fallbacks TEXT,                 -- comma-joined
    finish_reason TEXT,
    prompt_ms REAL, predicted_ms REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    checks_json TEXT NOT NULL,      -- [{name, passed, detail}, ...]
    PRIMARY KEY (run_id, fixture_id, sample_idx, stage)
);

-- Judge output over a (ticker, as_of) decision. Keyed on judge_model too, so
-- re-judging with a different/bigger model doesn't clobber a prior judgment.
-- Superseded by criterion_verdicts (one row per criterion, no imputed score),
-- but deliberately left in place with its rows: it holds the first judged batch.
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

-- One row per (decision, judge, criterion). `verdict` is NULL when the judge's
-- response was unparseable — that is a fact worth storing, and the reason this
-- table exists rather than a wider one: a NULL here cannot be silently averaged
-- as a middling score the way an imputed 5.0 was.
CREATE TABLE IF NOT EXISTS criterion_verdicts (
    ticker TEXT, as_of TEXT, model TEXT, judge_model TEXT,
    criterion TEXT, verdict TEXT, reason TEXT, judged_at TEXT,
    PRIMARY KEY (ticker, as_of, model, judge_model, criterion)
);

-- Pairwise comparisons. order_shown is in the PK because the same pair MUST be
-- judged both ways: the disagreement rate between the two orders is the judge's
-- own noise floor, and storing only one order throws that measurement away.
CREATE TABLE IF NOT EXISTS pairwise_results (
    ticker TEXT, as_of_a TEXT, as_of_b TEXT,
    model_a TEXT, model_b TEXT, judge_model TEXT,
    criterion TEXT, order_shown TEXT, winner TEXT, reason TEXT, judged_at TEXT,
    PRIMARY KEY (ticker, as_of_a, as_of_b, judge_model, criterion, order_shown)
);
"""


# Columns that were added to a table after that table already existed in the
# wild. `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so it can
# never add these — _migrate does.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "decisions": [("model", "TEXT")],
    "debate_turns": [("model", "TEXT")],
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive-only column adds for DBs created before a column existed.

    Idempotent: checks PRAGMA table_info before each ALTER, so it is safe to
    run on every connection. Never drops or rewrites a table — data/edge_analyst.db
    holds the 2026-07-23 live baseline. Deliberately does not touch data: a
    migration must not guess at what an existing NULL should have been (see
    eval/backfill_model_column.py for that, as a one-shot opt-in).
    """
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, column_type in columns:
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    conn.commit()


def get_connection(db_path: str | Path = "data/edge_analyst.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    _migrate(conn)
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


def _free_as_of(
    conn: sqlite3.Connection, ticker: str, as_of: str, model: str | None
) -> str:
    """Resolve an `as_of` that won't clobber another model's decision.

    The PK stays (ticker, as_of) — rebuilding it to include `model` means a full
    table copy in SQLite, which is not worth risking against the live baseline
    DB. Instead: two models producing a decision for the same ticker inside the
    same second (as_of has second resolution) would collide, so suffix rather
    than INSERT OR REPLACE over the other model's row. Same model re-running is
    still an idempotent overwrite, which is the behaviour we want.
    """
    candidate = as_of
    suffix = 1
    while True:
        row = conn.execute(
            "SELECT model FROM decisions WHERE ticker = ? AND as_of = ?",
            (ticker, candidate),
        ).fetchone()
        if row is None or row[0] == model:
            return candidate
        suffix += 1
        candidate = f"{as_of}#{suffix}"


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
    model: str | None = None,
) -> str:
    """Returns the `as_of` actually written — may differ from the one passed in
    when another model already holds that key (see _free_as_of). Callers must
    use the returned value to key related rows, e.g. save_debate_turns."""
    as_of = _free_as_of(conn, ticker, as_of, model)
    conn.execute(
        """INSERT OR REPLACE INTO decisions
           (ticker, as_of, close, rsi, macd_hist, gate_reasons, news_text,
            sentiment_label, sentiment_score, sentiment_confidence, sentiment_rationale,
            trader_action, trader_reasoning,
            trader_entry_price, trader_stop_loss, trader_position_sizing, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            model,
        ),
    )
    conn.commit()
    return as_of


def save_debate_turns(
    conn: sqlite3.Connection,
    ticker: str,
    as_of: str,
    history: list[DebateState],
    model: str | None = None,
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
                model,
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
                model,
            )
        )
    conn.executemany(
        """INSERT OR REPLACE INTO debate_turns
           (ticker, as_of, round, side, stance, key_point, confidence, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
    "model",
]


def _nest_debate_turns(conn: sqlite3.Connection, records: list[dict]) -> list[dict]:
    """Attaches each record's debate history under "debate_turns" — everything
    eval/rubric.py's prompt builders need, no further DB access from the judge."""
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


def fetch_decisions_for_judging(
    conn: sqlite3.Connection, limit: int = 20, model: str | None = None
) -> list[dict]:
    """One dict per (ticker, as_of) decision, most recent first, with its full
    debate history nested. `model` narrows to a single model's decisions — the
    only way to judge one model's output without another's rows mixed in."""
    columns = ", ".join(_DECISION_COLUMNS)
    if model is None:
        rows = conn.execute(
            f"SELECT {columns} FROM decisions ORDER BY as_of DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT {columns} FROM decisions WHERE model = ?
                ORDER BY as_of DESC LIMIT ?""",
            (model, limit),
        ).fetchall()
    records = [dict(zip(_DECISION_COLUMNS, row, strict=True)) for row in rows]
    return _nest_debate_turns(conn, records)


def fetch_decisions_for_pairwise(
    conn: sqlite3.Connection, model_a: str, model_b: str
) -> list[tuple[dict, dict]]:
    """Decisions from two models paired on (ticker, calendar day).

    Pairing on the day rather than the exact `as_of` is deliberate: two models
    run back to back never share a second-resolution timestamp, but comparing
    them only makes sense on the same ticker and the same day's indicators.
    Where a model has several decisions for one ticker-day, the most recent is
    taken, so a re-run supersedes rather than multiplying the pairs.
    """
    columns = ", ".join(_DECISION_COLUMNS)
    rows = conn.execute(
        f"""SELECT {columns} FROM decisions
            WHERE model IN (?, ?) ORDER BY as_of ASC""",
        (model_a, model_b),
    ).fetchall()
    records = _nest_debate_turns(
        conn, [dict(zip(_DECISION_COLUMNS, row, strict=True)) for row in rows]
    )

    # Later rows overwrite earlier ones for the same key (rows are as_of ASC).
    by_key: dict[tuple[str, str, str], dict] = {}
    for record in records:
        day = (record["as_of"] or "")[:10]
        by_key[(record["ticker"], day, record["model"])] = record

    pairs = []
    for (ticker, day, model), record in by_key.items():
        if model != model_a:
            continue
        counterpart = by_key.get((ticker, day, model_b))
        if counterpart is not None:
            pairs.append((record, counterpart))
    return pairs


_EVAL_SAMPLE_COLUMNS = [
    "run_id",
    "fixture_id",
    "sample_idx",
    "stage",
    "raw_output",
    "parsed_json",
    "fallbacks",
    "finish_reason",
    "prompt_ms",
    "predicted_ms",
    "prompt_tokens",
    "completion_tokens",
    "checks_json",
]


def save_eval_run(conn: sqlite3.Connection, run: dict) -> None:
    """Upsert one Tier 0 run's metadata. Called twice — once at the start so a
    crashed run still leaves a record of what was attempted, once at the end to
    stamp finished_at."""
    conn.execute(
        """INSERT OR REPLACE INTO eval_runs
           (run_id, model_name, base_url, stage, k, seed, temperature,
            started_at, finished_at, git_sha)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run["run_id"],
            run["model_name"],
            run.get("base_url"),
            run.get("stage"),
            run.get("k"),
            run.get("seed"),
            run.get("temperature"),
            run.get("started_at"),
            run.get("finished_at"),
            run.get("git_sha"),
        ),
    )
    conn.commit()


def save_eval_sample(conn: sqlite3.Connection, sample: dict) -> None:
    conn.execute(
        f"""INSERT OR REPLACE INTO eval_samples
            ({", ".join(_EVAL_SAMPLE_COLUMNS)})
            VALUES ({", ".join("?" * len(_EVAL_SAMPLE_COLUMNS))})""",
        tuple(sample.get(column) for column in _EVAL_SAMPLE_COLUMNS),
    )
    conn.commit()


def fetch_eval_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    columns = [
        "run_id",
        "model_name",
        "base_url",
        "stage",
        "k",
        "seed",
        "temperature",
        "started_at",
        "finished_at",
        "git_sha",
    ]
    row = conn.execute(
        f"SELECT {', '.join(columns)} FROM eval_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return dict(zip(columns, row, strict=True)) if row else None


def fetch_eval_samples(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        f"""SELECT {", ".join(_EVAL_SAMPLE_COLUMNS)} FROM eval_samples
            WHERE run_id = ? ORDER BY fixture_id, stage, sample_idx""",
        (run_id,),
    ).fetchall()
    return [dict(zip(_EVAL_SAMPLE_COLUMNS, row, strict=True)) for row in rows]


def latest_eval_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM eval_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


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


_CRITERION_COLUMNS = [
    "ticker",
    "as_of",
    "model",
    "judge_model",
    "criterion",
    "verdict",
    "reason",
    "judged_at",
]


def save_criterion_verdicts(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """One row per (decision, judge, criterion). A NULL `verdict` means the
    judge's response was unparseable and is stored as such — see the table
    comment for why that must not become a default."""
    conn.executemany(
        f"""INSERT OR REPLACE INTO criterion_verdicts
            ({", ".join(_CRITERION_COLUMNS)})
            VALUES ({", ".join("?" * len(_CRITERION_COLUMNS))})""",
        [tuple(row.get(column) for column in _CRITERION_COLUMNS) for row in rows],
    )
    conn.commit()


def fetch_criterion_verdicts(
    conn: sqlite3.Connection, judge_model: str | None = None, model: str | None = None
) -> list[dict]:
    where, params = [], []
    if judge_model is not None:
        where.append("judge_model = ?")
        params.append(judge_model)
    if model is not None:
        where.append("model = ?")
        params.append(model)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""SELECT {", ".join(_CRITERION_COLUMNS)} FROM criterion_verdicts
            {clause} ORDER BY ticker, as_of, criterion""",
        tuple(params),
    ).fetchall()
    return [dict(zip(_CRITERION_COLUMNS, row, strict=True)) for row in rows]


_PAIRWISE_COLUMNS = [
    "ticker",
    "as_of_a",
    "as_of_b",
    "model_a",
    "model_b",
    "judge_model",
    "criterion",
    "order_shown",
    "winner",
    "reason",
    "judged_at",
]


def save_pairwise_results(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        f"""INSERT OR REPLACE INTO pairwise_results
            ({", ".join(_PAIRWISE_COLUMNS)})
            VALUES ({", ".join("?" * len(_PAIRWISE_COLUMNS))})""",
        [tuple(row.get(column) for column in _PAIRWISE_COLUMNS) for row in rows],
    )
    conn.commit()


def fetch_pairwise_results(
    conn: sqlite3.Connection,
    model_a: str | None = None,
    model_b: str | None = None,
    judge_model: str | None = None,
) -> list[dict]:
    where, params = [], []
    for column, value in (
        ("model_a", model_a),
        ("model_b", model_b),
        ("judge_model", judge_model),
    ):
        if value is not None:
            where.append(f"{column} = ?")
            params.append(value)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""SELECT {", ".join(_PAIRWISE_COLUMNS)} FROM pairwise_results
            {clause} ORDER BY ticker, as_of_a, criterion, order_shown""",
        tuple(params),
    ).fetchall()
    return [dict(zip(_PAIRWISE_COLUMNS, row, strict=True)) for row in rows]
