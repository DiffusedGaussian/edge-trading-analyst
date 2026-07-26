"""Smoke tests over the news/sentiment analyst prompt + parser.

Pins the forgiving-parser contract: every field falls back to a safe default
on a miss rather than raising, and the score is clamped to [0, 10].
"""

from __future__ import annotations

from edge_analyst.news_analyst import build_sentiment_prompt, parse_sentiment_response


def test_build_sentiment_prompt_anchors_on_real_indicators():
    messages = build_sentiment_prompt(
        "AAPL", 150.0, 25.0, -0.5, ["rsi_oversold"], "Some headline."
    )
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "AAPL" in user_content
    assert "25.0" in user_content
    assert "rsi_oversold" in user_content
    assert "Some headline." in user_content


def test_parse_sentiment_response_valid_input():
    text = "LABEL: bullish\nSCORE: 7\nCONFIDENCE: high\nRATIONALE: RSI is oversold."
    signal = parse_sentiment_response(text)
    assert signal.label == "bullish"
    assert signal.score == 7.0
    assert signal.confidence == "high"
    assert signal.rationale == "RSI is oversold."


def test_parse_sentiment_response_missing_fields_fall_back_to_defaults():
    signal = parse_sentiment_response("some unparseable prose with no fields")
    assert signal.label == "neutral"
    assert signal.confidence == "low"
    assert signal.score == 5.0
    assert signal.rationale == "no rationale parsed"


def test_parse_sentiment_response_clamps_out_of_range_score():
    assert parse_sentiment_response("LABEL: bullish\nSCORE: 99").score == 10.0
    assert parse_sentiment_response("LABEL: bullish\nSCORE: -5").score == 0.0


def test_fallbacks_empty_on_well_formed_output():
    text = "LABEL: bullish\nSCORE: 7\nCONFIDENCE: high\nRATIONALE: RSI is oversold."
    assert parse_sentiment_response(text).fallbacks == frozenset()


def test_fallbacks_names_every_defaulted_field():
    signal = parse_sentiment_response("some unparseable prose with no fields")
    assert signal.fallbacks == frozenset({"LABEL", "SCORE", "CONFIDENCE", "RATIONALE"})


def test_fallbacks_flags_garbage_score_the_same_as_a_missing_one():
    """Both produce 5.0 at runtime, and 5.0 must never read as a real score."""
    garbage = parse_sentiment_response(
        "LABEL: bullish\nSCORE: abc\nCONFIDENCE: high\nRATIONALE: a fact."
    )
    missing = parse_sentiment_response(
        "LABEL: bullish\nCONFIDENCE: high\nRATIONALE: a fact."
    )
    assert garbage.score == missing.score == 5.0
    assert garbage.fallbacks == missing.fallbacks == frozenset({"SCORE"})


def test_fallbacks_flags_an_out_of_vocabulary_label():
    """A model answering "LABEL: very bullish" parsed to neutral — that is a
    format failure, not a neutral read of the market."""
    signal = parse_sentiment_response(
        "LABEL: very bullish\nSCORE: 8\nCONFIDENCE: high\nRATIONALE: a fact."
    )
    assert signal.label == "neutral"
    assert signal.fallbacks == frozenset({"LABEL"})


def test_fallbacks_flags_an_empty_rationale_value():
    signal = parse_sentiment_response(
        "LABEL: bullish\nSCORE: 7\nCONFIDENCE: high\nRATIONALE:   "
    )
    assert signal.rationale == "no rationale parsed"
    assert signal.fallbacks == frozenset({"RATIONALE"})
