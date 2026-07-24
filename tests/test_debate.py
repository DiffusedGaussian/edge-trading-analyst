"""Smoke tests over the bull/bear debate + trader synthesis.

Pins: the fixed-size DebateState (overwrite-not-append) contract, the
buy-vs-sell-standoff continuation rule, the forgiving parser defaults shared
with news_analyst.py, and that run_debate's returned history has one entry
per round actually run.
"""

from __future__ import annotations

from edge_analyst.debate import (
    DebateState,
    DebateTurn,
    build_debate_prompt,
    build_trader_prompt,
    parse_debate_response,
    parse_trader_response,
    run_debate,
    run_trader,
    should_continue_debate,
)


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _mock_replies(monkeypatch, replies: list[str]):
    """Feeds canned model replies in call order via llama-server's HTTP
    boundary — the one point every caller funnels through regardless of
    whether it imported chat_completion at module level or per-call."""
    it = iter(replies)

    def fake_post(url, json, timeout):
        return _FakeResponse(next(it))

    monkeypatch.setattr("edge_analyst.llm_client.requests.post", fake_post)


def test_build_debate_prompt_selects_persona_and_includes_rebuttal():
    bull_messages = build_debate_prompt(
        "bull", "AAPL", 150.0, 55.0, 0.5, [], "news", "Bear's point"
    )
    assert "Bull analyst" in bull_messages[0]["content"]
    assert "Bear's current strongest point: Bear's point" in bull_messages[1]["content"]

    bear_messages = build_debate_prompt(
        "bear", "AAPL", 150.0, 55.0, 0.5, [], "news", None
    )
    assert "Bear analyst" in bear_messages[0]["content"]
    assert "strongest point" not in bear_messages[1]["content"]


def test_parse_debate_response_defaults_on_miss():
    turn = parse_debate_response("unparseable prose")
    assert turn.stance == "hold"
    assert turn.confidence == "low"
    assert turn.key_point == "no key point parsed"


def test_should_continue_debate_only_on_buy_sell_standoff():
    bull = DebateTurn(stance="buy", key_point="x", confidence="high")
    bear = DebateTurn(stance="sell", key_point="y", confidence="high")
    assert should_continue_debate(DebateState(round=1, bull=bull, bear=bear)) is True

    bear_hold = DebateTurn(stance="hold", key_point="y", confidence="low")
    assert (
        should_continue_debate(DebateState(round=1, bull=bull, bear=bear_hold)) is False
    )


def test_build_trader_prompt_includes_both_sides():
    bull = DebateTurn(stance="buy", key_point="strong earnings", confidence="high")
    bear = DebateTurn(stance="sell", key_point="overbought", confidence="low")
    state = DebateState(round=1, bull=bull, bear=bear)
    messages = build_trader_prompt("AAPL", 150.0, 55.0, 0.5, [], state)
    assert "strong earnings" in messages[1]["content"]
    assert "overbought" in messages[1]["content"]


def test_parse_trader_response_na_fields_become_none():
    text = (
        "ACTION: Hold\nREASONING: mixed signals.\n"
        "ENTRY_PRICE: NA\nSTOP_LOSS: NA\nPOSITION_SIZING: NA"
    )
    decision = parse_trader_response(text)
    assert decision.action == "hold"
    assert decision.entry_price is None
    assert decision.stop_loss is None
    assert decision.position_sizing is None


def test_parse_trader_response_valid_numbers():
    text = (
        "ACTION: Buy\nREASONING: strong momentum.\n"
        "ENTRY_PRICE: 150.5\nSTOP_LOSS: 140.0\nPOSITION_SIZING: 5"
    )
    decision = parse_trader_response(text)
    assert decision.action == "buy"
    assert decision.entry_price == 150.5
    assert decision.stop_loss == 140.0
    assert decision.position_sizing == 5.0


def test_run_debate_stops_early_on_convergence(monkeypatch):
    # Round 1: bull=buy, bear=hold -> not a standoff -> stop after 1 round.
    _mock_replies(
        monkeypatch,
        [
            "STANCE: Buy\nKEY_POINT: strong earnings\nCONFIDENCE: high",
            "STANCE: Hold\nKEY_POINT: mixed signals\nCONFIDENCE: low",
        ],
    )
    final, history = run_debate("AAPL", 150.0, 55.0, 0.5, [], "news", "http://x")
    assert len(history) == 1
    assert final.round == 1
    assert final.bull.stance == "buy"
    assert final.bear.stance == "hold"


def test_run_debate_runs_to_max_rounds_on_standoff(monkeypatch):
    # Both rounds are a buy-vs-sell standoff -> runs until max_rounds (2).
    _mock_replies(
        monkeypatch,
        [
            "STANCE: Buy\nKEY_POINT: a\nCONFIDENCE: high",
            "STANCE: Sell\nKEY_POINT: b\nCONFIDENCE: high",
            "STANCE: Buy\nKEY_POINT: c\nCONFIDENCE: high",
            "STANCE: Sell\nKEY_POINT: d\nCONFIDENCE: high",
        ],
    )
    final, history = run_debate("AAPL", 150.0, 55.0, 0.5, [], "news", "http://x")
    assert len(history) == 2
    assert final.round == 2


def test_run_trader_parses_final_decision(monkeypatch):
    _mock_replies(
        monkeypatch,
        [
            "ACTION: Buy\nREASONING: strong case.\n"
            "ENTRY_PRICE: 150\nSTOP_LOSS: 140\nPOSITION_SIZING: 5"
        ],
    )
    bull = DebateTurn(stance="buy", key_point="a", confidence="high")
    bear = DebateTurn(stance="hold", key_point="b", confidence="low")
    state = DebateState(round=1, bull=bull, bear=bear)
    decision = run_trader("AAPL", 150.0, 55.0, 0.5, [], state, "http://x")
    assert decision.action == "buy"
    assert decision.entry_price == 150.0
