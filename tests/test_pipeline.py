"""Integration tests over run_cycle's early-exit ladder and the full LLM
cascade + persistence path. All external boundaries are synthetic/mocked:
data_source (no network) and llama-server's HTTP endpoint (no model).
"""

from __future__ import annotations

import pandas as pd

from edge_analyst import data_source, pipeline, store


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _mock_replies(monkeypatch, replies: list[str]):
    it = iter(replies)

    def fake_post(url, json, timeout):
        return _FakeResponse(next(it))

    monkeypatch.setattr("edge_analyst.llm_client.requests.post", fake_post)


def _synthetic_ohlcv(
    n_days: int = 60, final_jump_pct: float | None = None
) -> pd.DataFrame:
    """A gentle, steady uptrend — day-over-day moves stay well under the 3%
    gate threshold, and any RSI/MACD crossings happen early in the warmup,
    not on the last row, so the gate reads quiet by construction. Passing
    final_jump_pct overrides the last close to deterministically fire the
    price_move rule instead."""
    closes = [100.0 + 0.3 * i for i in range(n_days)]
    if final_jump_pct is not None:
        closes[-1] = closes[-2] * (1 + final_jump_pct)
    index = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "open": [c - 0.05 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n_days,
        },
        index=index,
    )


_FUNDAMENTALS = {
    "trailing_pe": 20.0,
    "forward_pe": 18.0,
    "market_cap": 1e12,
    "beta": 1.1,
}


def test_run_cycle_no_data_persists_nothing(monkeypatch):
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: pd.DataFrame())
    conn = store.get_connection(":memory:")

    result = pipeline.run_cycle(conn, "AAPL", 60)

    assert not result.snapshot.has_data
    assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 0


def test_run_cycle_quiet_gate_persists_core_but_skips_cascade(monkeypatch):
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: _synthetic_ohlcv())
    monkeypatch.setattr(
        data_source, "fetch_fundamentals", lambda *a, **k: _FUNDAMENTALS
    )
    conn = store.get_connection(":memory:")

    result = pipeline.run_cycle(conn, "AAPL", 60)

    assert result.snapshot.has_data
    assert result.snapshot.gate_result.material is False
    assert result.sentiment is None
    assert result.debate is None
    assert result.trader is None
    assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 60
    assert conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_run_cycle_material_gate_without_base_url_skips_cascade(monkeypatch):
    monkeypatch.setattr(
        data_source,
        "fetch_ohlcv",
        lambda *a, **k: _synthetic_ohlcv(final_jump_pct=0.10),
    )
    monkeypatch.setattr(
        data_source, "fetch_fundamentals", lambda *a, **k: _FUNDAMENTALS
    )
    conn = store.get_connection(":memory:")

    result = pipeline.run_cycle(conn, "AAPL", 60)

    assert result.snapshot.gate_result.material is True
    assert "price_move" in result.snapshot.gate_result.reasons
    assert result.sentiment is None
    assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 60
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_run_cycle_quiet_gate_with_force_runs_full_cascade_and_persists(monkeypatch):
    monkeypatch.setattr(data_source, "fetch_ohlcv", lambda *a, **k: _synthetic_ohlcv())
    monkeypatch.setattr(
        data_source, "fetch_fundamentals", lambda *a, **k: _FUNDAMENTALS
    )
    monkeypatch.setattr(data_source, "fetch_news", lambda *a, **k: "Some headline.")
    # analyst -> bull(round1) -> bear(round1, converges to Hold) -> trader.
    _mock_replies(
        monkeypatch,
        [
            "LABEL: bullish\nSCORE: 7\nCONFIDENCE: high\nRATIONALE: fact.",
            "STANCE: Buy\nKEY_POINT: a\nCONFIDENCE: high",
            "STANCE: Hold\nKEY_POINT: b\nCONFIDENCE: low",
            "ACTION: Hold\nREASONING: mixed.\n"
            "ENTRY_PRICE: NA\nSTOP_LOSS: NA\nPOSITION_SIZING: NA",
        ],
    )
    conn = store.get_connection(":memory:")

    result = pipeline.run_cycle(conn, "AAPL", 60, base_url="http://x", force=True)

    assert result.sentiment.label == "bullish"
    assert result.trader.action == "hold"
    assert len(result.debate_history) == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM debate_turns").fetchone()[0] == 2


def test_run_cycle_material_gate_with_base_url_runs_full_cascade(monkeypatch):
    monkeypatch.setattr(
        data_source,
        "fetch_ohlcv",
        lambda *a, **k: _synthetic_ohlcv(final_jump_pct=0.10),
    )
    monkeypatch.setattr(
        data_source, "fetch_fundamentals", lambda *a, **k: _FUNDAMENTALS
    )
    monkeypatch.setattr(data_source, "fetch_news", lambda *a, **k: "Some headline.")
    # A genuine buy-vs-sell standoff runs to max_rounds (2).
    _mock_replies(
        monkeypatch,
        [
            "LABEL: bullish\nSCORE: 8\nCONFIDENCE: high\nRATIONALE: fact.",
            "STANCE: Buy\nKEY_POINT: a\nCONFIDENCE: high",
            "STANCE: Sell\nKEY_POINT: b\nCONFIDENCE: high",
            "STANCE: Buy\nKEY_POINT: c\nCONFIDENCE: high",
            "STANCE: Sell\nKEY_POINT: d\nCONFIDENCE: high",
            "ACTION: Buy\nREASONING: strong case.\n"
            "ENTRY_PRICE: 110\nSTOP_LOSS: 100\nPOSITION_SIZING: 5",
        ],
    )
    conn = store.get_connection(":memory:")

    result = pipeline.run_cycle(conn, "AAPL", 60, base_url="http://x")

    assert len(result.debate_history) == 2
    assert result.trader.action == "buy"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    turns = conn.execute(
        "SELECT round, side FROM debate_turns ORDER BY round, side"
    ).fetchall()
    assert turns == [(1, "bear"), (1, "bull"), (2, "bear"), (2, "bull")]
