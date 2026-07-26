"""Tests over the Tier 1 judge's prompt builders and parsers (eval/rubric.py).

Pure functions only — no Modal, no vLLM. Rewritten from the single four-question
rubric to the per-criterion and pairwise APIs. The load-bearing assertions are
the ones about *not* imputing: an unparseable judge response must produce None,
because a run where the judge often failed to answer is invalid rather than
low-scoring, and a default erases that distinction permanently.
"""

from __future__ import annotations

import pytest
from eval.rubric import (
    CRITERIA,
    CRITERION_NAMES,
    ORDERS,
    PairwiseVerdict,
    build_judge_prompt,
    build_pairwise_prompt,
    format_record,
    model_family,
    order_flip,
    parse_pairwise,
    parse_verdict,
    resolve_pairwise_winner,
    same_family,
)

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
    "model": "gemma-3-1b-it",
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

_RECORD_B = {
    **_RECORD,
    "as_of": "2026-07-24T14:00:00",
    "model": "olmoe-1b-7b",
    "trader_reasoning": "a distinctly different rationale",
    "sentiment_rationale": "another fact entirely",
}


# --- per-criterion prompts ---------------------------------------------------


def test_four_criteria_each_with_its_own_prompt():
    assert CRITERION_NAMES == (
        "bull_bear_distinct",
        "indicator_consistent",
        "news_fidelity",
        "trader_consistent",
    )
    assert len({CRITERIA[name] for name in CRITERION_NAMES}) == 4


def test_every_criterion_prompt_has_a_worked_yes_and_no_example():
    """Without them each criterion drifts toward "is this good analysis overall",
    and four independent questions collapse into one."""
    for name, prompt in CRITERIA.items():
        assert "Example (yes):" in prompt, name
        assert "Example (no):" in prompt, name
        assert "VERDICT: yes" in prompt, name
        assert "VERDICT: no" in prompt, name


def test_no_criterion_prompt_asks_for_an_overall_score():
    """An LLM's absolute 0-10 clusters at 7-8; report.py derives yes/answered."""
    for name, prompt in CRITERIA.items():
        assert "OVERALL_SCORE" not in prompt, name


def test_build_judge_prompt_includes_all_record_fields():
    messages = build_judge_prompt(_RECORD, "indicator_consistent")
    user_content = messages[1]["content"]
    assert "AAPL" in user_content
    assert "price_move" in user_content
    assert "Some headline." in user_content
    assert "bullish" in user_content
    assert "strong earnings" in user_content
    assert "overbought" in user_content
    assert "strong case" in user_content


def test_build_judge_prompt_asks_only_its_own_criterion():
    messages = build_judge_prompt(_RECORD, "news_fidelity")
    system = messages[0]["content"]
    assert "news" in system.lower()
    assert "bull and the bear" not in system


def test_build_judge_prompt_rejects_an_unknown_criterion():
    with pytest.raises(KeyError):
        build_judge_prompt(_RECORD, "vibes")


def test_format_record_handles_no_debate_turns():
    record = {**_RECORD, "debate_turns": []}
    assert "No debate rounds recorded." in format_record(record)


# --- verdict parsing --------------------------------------------------------


def test_parse_verdict_reads_yes_and_no():
    yes = parse_verdict("VERDICT: yes\nREASON: it matches.", "news_fidelity")
    assert yes.verdict == "yes"
    assert yes.reason == "it matches."

    no = parse_verdict("VERDICT: no\nREASON: it contradicts.", "news_fidelity")
    assert no.verdict == "no"


def test_parse_verdict_is_case_and_punctuation_tolerant():
    for text in ("VERDICT: Yes", "verdict: YES.", "VERDICT: yes, clearly"):
        assert parse_verdict(text, "news_fidelity").verdict == "yes"


def test_parse_verdict_never_imputes_a_default():
    """The point of the rewrite: an unparseable judge response is unknown, not a
    middling score. A default made a bad run indistinguishable from an invalid
    one, permanently, because the imputed value was then averaged in."""
    for text in ("", "I'm not sure about this one.", "VERDICT: maybe"):
        assert parse_verdict(text, "news_fidelity").verdict is None, text


def test_parse_verdict_keeps_a_reason_even_when_the_verdict_is_unparseable():
    parsed = parse_verdict("VERDICT: unclear\nREASON: the record is ambiguous.", "x")
    assert parsed.verdict is None
    assert parsed.reason == "the record is ambiguous."


# --- pairwise ---------------------------------------------------------------


def test_build_pairwise_prompt_ab_order_shows_a_first():
    messages = build_pairwise_prompt(_RECORD, _RECORD_B, "news_fidelity", "ab")
    first_block = messages[1]["content"].split("=== RECORD B ===")[0]
    assert "a fact" in first_block
    assert "another fact entirely" not in first_block


def test_build_pairwise_prompt_ba_order_genuinely_swaps_the_records():
    """If this ever stops swapping, order_flip_rate silently measures nothing and
    the judge's position bias becomes invisible."""
    messages = build_pairwise_prompt(_RECORD, _RECORD_B, "news_fidelity", "ba")
    first_block = messages[1]["content"].split("=== RECORD B ===")[0]
    assert "another fact entirely" in first_block
    assert "a fact" not in first_block


def test_build_pairwise_prompt_validates_order_and_criterion():
    with pytest.raises(ValueError):
        build_pairwise_prompt(_RECORD, _RECORD_B, "news_fidelity", "ba2")
    with pytest.raises(KeyError):
        build_pairwise_prompt(_RECORD, _RECORD_B, "vibes", "ab")


def test_build_pairwise_prompt_allows_a_tie():
    system = build_pairwise_prompt(_RECORD, _RECORD_B, "news_fidelity", "ab")[0]
    assert "tie" in system["content"]


def test_parse_pairwise_reads_a_b_and_tie():
    assert parse_pairwise("WINNER: A\nREASON: x", "c", "ab").winner == "A"
    assert parse_pairwise("WINNER: b\nREASON: x", "c", "ab").winner == "B"
    assert parse_pairwise("WINNER: tie\nREASON: x", "c", "ab").winner == "tie"
    assert parse_pairwise("WINNER: Record B\nREASON: x", "c", "ab").winner == "B"


def test_parse_pairwise_never_imputes():
    for text in ("", "Both are fine really.", "WINNER: neither"):
        assert parse_pairwise(text, "c", "ab").winner is None, text


def test_resolve_pairwise_winner_undoes_the_display_order():
    """Position A in "ba" order is record B — get this backwards and the
    comparison reports the loser as the winner."""
    assert resolve_pairwise_winner(PairwiseVerdict("c", "ab", "A", None)) == "model_a"
    assert resolve_pairwise_winner(PairwiseVerdict("c", "ba", "A", None)) == "model_b"
    assert resolve_pairwise_winner(PairwiseVerdict("c", "ba", "B", None)) == "model_a"
    assert resolve_pairwise_winner(PairwiseVerdict("c", "ab", "tie", None)) == "tie"
    assert resolve_pairwise_winner(PairwiseVerdict("c", "ab", None, None)) is None


def test_order_flip_detects_a_position_bias():
    # The judge picked whichever record was shown first: a pure position bias.
    assert (
        order_flip(
            PairwiseVerdict("c", "ab", "A", None),
            PairwiseVerdict("c", "ba", "A", None),
        )
        is True
    )
    # The judge picked the same *record* in both orders: a real preference.
    assert (
        order_flip(
            PairwiseVerdict("c", "ab", "A", None),
            PairwiseVerdict("c", "ba", "B", None),
        )
        is False
    )


def test_order_flip_is_unknown_when_either_direction_failed_to_parse():
    assert (
        order_flip(
            PairwiseVerdict("c", "ab", "A", None),
            PairwiseVerdict("c", "ba", None, None),
        )
        is None
    )


def test_orders_covers_both_directions():
    assert set(ORDERS) == {"ab", "ba"}


# --- judge-family bias ------------------------------------------------------


def test_same_family_flags_a_qwen_judge_scoring_qwen_output():
    assert same_family("Qwen3-30B-A3B", "Qwen/Qwen2.5-32B-Instruct")


def test_same_family_is_false_across_families():
    assert not same_family("olmoe-1b-7b", "Qwen/Qwen2.5-32B-Instruct")
    assert not same_family("gemma-3-1b-it", "mistralai/Mistral-Small-24B-Instruct-2501")


def test_model_family_returns_none_for_an_unrecognised_name():
    assert model_family("some-local-gguf") is None
    assert not same_family("some-local-gguf", "another-local-gguf")
