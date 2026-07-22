"""SQLite persistence for OHLCV + indicators + fundamentals snapshots.
Plain stdlib sqlite3 for Phase 1 — swap for Turso/libSQL later (Phase 3+)
once we need native vector search for the semantic cache; the schema below
is designed to migrate cleanly since it's just relational tables.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

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
            row.open, row.high, row.low, row.close, int(row.volume),
            row.macd, row.macd_signal, row.macd_hist, row.rsi,
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


def save_fundamentals(conn: sqlite3.Connection, ticker: str, as_of: str, f: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO fundamentals
           (ticker, as_of, trailing_pe, forward_pe, market_cap, beta)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ticker, as_of, f.get("trailing_pe"), f.get("forward_pe"),
         f.get("market_cap"), f.get("beta")),
    )
    conn.commit()
