"""Smoke tests over the LLM-judge rubric prompt + parser (eval/rubric.py).

Pure functions only — no Modal/vLLM here. Pins the forgiving-parser contract
shared with news_analyst.py/debate.py: every field falls back to a safe
default on a miss rather than raising.
"""

from __future__ import annotations

from eval.rubric import build_judge_prompt, parse_judgment

_RECORD = {
    "ticker": "AAPL",
    "as_of": "2026-07-24T10:00:00",
    "close": 150.0,
    "rsi": 55.0,
    "macd_hist": 0.5,
    "gate_reasons": "price_move,rsi_oversold",
    "news_text": "Some headline.",
    "sentiment_label": "bullish",
    "sentiment_score": 7.0,
    "sentiment_confidence": "high",
    "sentiment_rationale": "a fact",
    "trader_action": "buy",
    "trader_reasoning": "strong case",
    "trader_entry_price": 150.0,
    "trader_stop_loss": 140.0,
    "trader_position_sizing": 5.0,
    "debate_turns": [
        {
            "round": 1,
            "side": "bull",
            "stance": "buy",
            "key_point": "strong earnings",
            "confidence": "high",
        },
        {
            "round": 1,
            "side": "bear",
            "stance": "sell",
            "key_point": "overbought",
            "confidence": "high",
        },
    ],
}


def test_build_judge_prompt_includes_all_record_fields():
    messages = build_judge_prompt(_RECORD)
    user_content = messages[1]["content"]
    assert "AAPL" in user_content
    assert "price_move" in user_content
    assert "Some headline." in user_content
    assert "bullish" in user_content
    assert "strong earnings" in user_content
    assert "overbought" in user_content
    assert "strong case" in user_content


def test_build_judge_prompt_handles_no_debate_turns():
    record = {**_RECORD, "debate_turns": []}
    messages = build_judge_prompt(record)
    assert "No debate rounds recorded." in messages[1]["content"]


def test_parse_judgment_valid_input():
    text = (
        "BULL_BEAR_DISTINCT: yes\nINDICATOR_CONSISTENT: no\n"
        "NEWS_FIDELITY: yes\nTRADER_CONSISTENT: yes\n"
        "OVERALL_SCORE: 6\n"
        "NOTES: Trader reasoning contradicts a bullish MACD histogram."
    )
    judgment = parse_judgment(text)
    assert judgment.bull_bear_distinct == "yes"
    assert judgment.indicator_consistent == "no"
    assert judgment.overall_score == 6.0
    assert "MACD" in judgment.notes


def test_parse_judgment_missing_fields_fall_back_to_defaults():
    judgment = parse_judgment("some unparseable prose with no fields")
    assert judgment.bull_bear_distinct == "unknown"
    assert judgment.indicator_consistent == "unknown"
    assert judgment.news_fidelity == "unknown"
    assert judgment.trader_consistent == "unknown"
    assert judgment.overall_score == 5.0
    assert judgment.notes == "no notes parsed"


def test_parse_judgment_clamps_out_of_range_score():
    assert parse_judgment("OVERALL_SCORE: 99").overall_score == 10.0
    assert parse_judgment("OVERALL_SCORE: -5").overall_score == 0.0
