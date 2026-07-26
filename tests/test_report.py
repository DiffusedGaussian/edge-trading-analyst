"""Tests over the scorecard's aggregation functions.

Driven by hand-built sample rows, so no run and no model is needed. The
degenerate cases carry most of the weight: an empty run and a k=1 run must
report `n/a`, never 0.0 — a fabricated zero reads as a measured result and is
the easiest way for a scorecard to lie.
"""

from __future__ import annotations

import json

from eval.report import (
    Stat,
    build_report,
    check_pass_rates,
    criterion_scores,
    fallback_rates,
    hard_failures,
    label_flip_rates,
    pairwise_summary,
    regressed_checks,
    render_comparison,
    render_criterion_scores,
    render_pairwise,
    render_rate,
    render_run,
    summarise,
    throughput,
    truncation_rate,
)


def _sample(
    *,
    fixture_id="probe",
    stage="sentiment",
    sample_idx=0,
    label="bullish",
    fallbacks="",
    finish_reason="stop",
    checks=(("fields_parsed", True, "ok"),),
    prompt_tokens=100,
    prompt_ms=200.0,
    completion_tokens=50,
    predicted_ms=1000.0,
) -> dict:
    return {
        "run_id": "RUN",
        "fixture_id": fixture_id,
        "stage": stage,
        "sample_idx": sample_idx,
        "raw_output": "...",
        "parsed_json": json.dumps({"label": label}),
        "fallbacks": fallbacks,
        "finish_reason": finish_reason,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "checks_json": json.dumps(
            [{"name": n, "passed": p, "detail": d} for n, p, d in checks]
        ),
    }


# --- Stat / summarise -------------------------------------------------------


def test_summarise_reports_mean_and_population_stdev():
    stat = summarise([10.0, 20.0, 30.0])
    assert stat.mean == 20.0
    assert stat.stdev is not None
    assert stat.n == 3


def test_summarise_of_one_value_has_undefined_stdev():
    """Not 0.0 — with one observation the spread is unknown."""
    stat = summarise([42.0])
    assert stat.mean == 42.0
    assert stat.stdev is None
    assert "n/a" in stat.render(" tok/s")


def test_summarise_of_nothing_is_entirely_unknown():
    stat = summarise([])
    assert stat == Stat(None, None, 0)
    assert stat.render() == "n/a"


def test_render_rate_distinguishes_zero_from_unknown():
    assert render_rate(0.0) == "0%"
    assert render_rate(None) == "n/a"


# --- fallback rates ---------------------------------------------------------


def test_fallback_rates_include_fields_that_never_fell_back():
    """The zeros have to be visible: a rate list built only from observed
    failures silently omits the fields that are fine."""
    samples = [_sample(fallbacks="SCORE"), _sample(fallbacks="")]
    rates = dict(fallback_rates(samples, "sentiment"))
    assert rates == {"SCORE": 0.5, "LABEL": 0.0, "CONFIDENCE": 0.0, "RATIONALE": 0.0}


def test_fallback_rates_are_ordered_worst_first():
    samples = [_sample(fallbacks="LABEL,SCORE"), _sample(fallbacks="SCORE")]
    assert [field for field, _ in fallback_rates(samples, "sentiment")][0] == "SCORE"


def test_fallback_rates_on_an_empty_run_are_unknown():
    assert all(rate is None for _, rate in fallback_rates([], "sentiment"))


# --- check pass rates -------------------------------------------------------


def test_check_pass_rates_count_only_checks_that_ran():
    """A check a fixture excluded must not be averaged in as a pass or a fail."""
    samples = [
        _sample(
            checks=(("fields_parsed", True, "ok"), ("news_grounding", False, "no"))
        ),
        _sample(checks=(("fields_parsed", False, "bad"),)),
    ]
    rates = {
        name: (passed, ran, rate)
        for name, passed, ran, rate in check_pass_rates(samples)
    }
    assert rates["fields_parsed"] == (1, 2, 0.5)
    assert rates["news_grounding"] == (0, 1, 0.0)


def test_check_pass_rates_are_ordered_worst_first():
    samples = [
        _sample(
            checks=(
                ("fields_parsed", True, "ok"),
                ("label_consistency", False, "bad"),
            )
        )
    ]
    assert check_pass_rates(samples)[0][0] == "label_consistency"


def test_check_pass_rates_on_an_empty_run_is_empty():
    assert check_pass_rates([]) == []


# --- truncation -------------------------------------------------------------


def test_truncation_rate():
    samples = [_sample(finish_reason="length"), _sample(finish_reason="stop")]
    assert truncation_rate(samples) == 0.5


def test_truncation_rate_of_an_empty_run_is_unknown():
    assert truncation_rate([]) is None


# --- label flips ------------------------------------------------------------


def test_label_flip_rate_counts_disagreement_with_the_modal_label():
    samples = [
        _sample(sample_idx=0, label="bullish"),
        _sample(sample_idx=1, label="bullish"),
        _sample(sample_idx=2, label="neutral"),
    ]
    assert label_flip_rates(samples, "sentiment") == [("probe", 1 / 3, "bullish")]


def test_label_flip_rate_is_zero_when_every_sample_agrees():
    samples = [_sample(sample_idx=i) for i in range(3)]
    assert label_flip_rates(samples, "sentiment")[0][1] == 0.0


def test_label_flip_rate_of_a_k1_run_is_zero_not_unknown():
    """With one sample there is nothing to disagree with — genuinely 0, since
    the modal label is the only label."""
    assert label_flip_rates([_sample()], "sentiment")[0][1] == 0.0


def test_label_flip_rate_reads_the_right_field_per_stage():
    row = _sample(stage="trader")
    row["parsed_json"] = json.dumps({"action": "hold"})
    assert label_flip_rates([row], "trader") == [("probe", 0.0, "hold")]


# --- throughput -------------------------------------------------------------


def test_throughput_derives_prefill_and_decode_separately():
    prefill, decode = throughput([_sample(prompt_tokens=100, prompt_ms=200.0)])
    assert prefill.mean == 500.0  # 100 tok / 0.2 s
    assert decode.mean == 50.0  # 50 tok / 1.0 s


def test_throughput_ignores_rows_without_llama_cpp_timings():
    """A server that doesn't send `timings` contributes nothing, not a zero that
    would drag the mean down."""
    samples = [
        _sample(prompt_tokens=100, prompt_ms=200.0),
        _sample(prompt_ms=None, predicted_ms=None),
    ]
    prefill, _ = throughput(samples)
    assert prefill.mean == 500.0
    assert prefill.n == 1


def test_throughput_of_an_empty_run_is_unknown():
    prefill, decode = throughput([])
    assert prefill.mean is None and decode.mean is None


# --- hard failures ----------------------------------------------------------


def test_hard_failures_deduplicate_across_samples():
    """The same failure on 5 samples is one problem, not five."""
    samples = [
        _sample(sample_idx=i, checks=(("label_consistency", False, "said overbought"),))
        for i in range(5)
    ]
    assert hard_failures(samples) == [
        ("probe", "sentiment", "label_consistency", "said overbought")
    ]


def test_hard_failures_omits_passing_checks():
    assert hard_failures([_sample()]) == []


# --- whole report -----------------------------------------------------------


_RUN = {
    "run_id": "RUN",
    "model_name": "test-model",
    "stage": "all",
    "k": 2,
    "seed": 1,
    "temperature": 0.0,
    "git_sha": "abc1234",
}


def test_build_report_groups_by_stage_in_fixed_order():
    samples = [_sample(stage="trader"), _sample(stage="sentiment")]
    report = build_report(_RUN, samples)
    assert list(report["stages"]) == ["sentiment", "trader"]


def test_render_run_handles_an_empty_run_without_dividing_by_zero():
    text = render_run(build_report(_RUN, []))
    assert "no samples recorded" in text


def test_render_run_includes_the_headline_numbers():
    samples = [
        _sample(fallbacks="SCORE", checks=(("fields_parsed", False, "fell back"),)),
        _sample(sample_idx=1),
    ]
    text = render_run(build_report(_RUN, samples))
    assert "fallback rate per sentinel field" in text
    assert "test-model" in text
    assert "abc1234" in text
    assert "hard failures" in text


def test_render_run_of_a_k1_run_reports_na_spread_not_zero():
    text = render_run(build_report({**_RUN, "k": 1}, [_sample()]))
    assert "+/- n/a" in text


# --- comparison -------------------------------------------------------------


def test_regressed_checks_finds_pass_to_fail():
    passing = build_report(_RUN, [_sample(checks=(("label_consistency", True, "ok"),))])
    failing = build_report(
        _RUN, [_sample(checks=(("label_consistency", False, "said overbought"),))]
    )
    assert regressed_checks(passing, failing) == [
        ("probe", "sentiment", "label_consistency")
    ]
    # And the reverse direction is not a regression.
    assert regressed_checks(failing, passing) == []


def test_regressed_checks_ignores_a_check_the_first_run_never_ran():
    """Absent is not the same as passed — otherwise adding a stage to run B
    reports every one of its failures as a regression."""
    without = build_report(_RUN, [_sample(stage="sentiment")])
    with_trader = build_report(
        _RUN,
        [
            _sample(stage="sentiment"),
            _sample(stage="trader", checks=(("fields_parsed", False, "bad"),)),
        ],
    )
    assert regressed_checks(without, with_trader) == []


def test_render_comparison_shows_a_delta_column_and_flags_regressions():
    a = build_report(
        {**_RUN, "run_id": "A", "model_name": "model-a"},
        [_sample(checks=(("label_consistency", True, "ok"),))],
    )
    b = build_report(
        {**_RUN, "run_id": "B", "model_name": "model-b"},
        [_sample(checks=(("label_consistency", False, "said overbought"),))],
    )
    text = render_comparison(a, b)
    assert "delta" in text
    assert "-100pp" in text
    assert "REGRESSED" in text


def test_render_comparison_warns_when_the_two_runs_used_different_settings():
    a = build_report({**_RUN, "k": 5}, [_sample()])
    b = build_report({**_RUN, "k": 1}, [_sample()])
    assert "WARNING" in render_comparison(a, b)


def test_render_comparison_of_two_empty_runs_does_not_raise():
    empty = build_report(_RUN, [])
    assert "no check regressed" in render_comparison(empty, empty)


# --- Tier 1: judge scores ---------------------------------------------------


def _verdict(criterion="news_fidelity", verdict="yes"):
    return {
        "ticker": "AAPL",
        "as_of": "2026-07-24T10:00:00",
        "model": "gemma-3-1b-it",
        "judge_model": "Qwen/Qwen2.5-32B-Instruct",
        "criterion": criterion,
        "verdict": verdict,
        "reason": "because",
    }


def test_derived_score_is_yes_over_answered_not_over_judged():
    """An unanswered criterion must shrink the denominator, not count as a no —
    otherwise a judge that fails to parse looks like a failing model."""
    verdicts = [
        _verdict(verdict="yes"),
        _verdict(verdict="no"),
        _verdict(verdict=None),
    ]
    scores = criterion_scores(verdicts)["news_fidelity"]
    assert scores["derived_score"] == 0.5
    assert scores["answered"] == 2
    assert scores["judged"] == 3


def test_parse_failure_rate_is_reported_separately_from_the_score():
    verdicts = [_verdict(verdict="yes"), _verdict(verdict=None)]
    scores = criterion_scores(verdicts)["news_fidelity"]
    assert scores["derived_score"] == 1.0
    assert scores["parse_failure_rate"] == 0.5


def test_derived_score_of_an_all_unparsed_criterion_is_unknown():
    """Not 0.0 — nothing was measured. This is the distinction the whole
    no-imputation rule exists to preserve."""
    scores = criterion_scores([_verdict(verdict=None)])["news_fidelity"]
    assert scores["derived_score"] is None
    assert scores["parse_failure_rate"] == 1.0


def test_render_criterion_scores_handles_no_verdicts():
    assert "no verdicts recorded" in render_criterion_scores([])


def test_render_criterion_scores_calls_a_high_parse_failure_run_invalid():
    verdicts = [_verdict(verdict=None), _verdict(verdict="yes")]
    text = render_criterion_scores(verdicts, "gemma-3-1b-it", "Qwen/Qwen2.5-32B")
    assert "invalid" in text


def test_render_criterion_scores_warns_on_a_same_family_judge():
    text = render_criterion_scores(
        [_verdict()], "Qwen3-30B-A3B", "Qwen/Qwen2.5-32B-Instruct"
    )
    assert "WARNING" in text
    assert "self-preference" in text


def test_render_criterion_scores_does_not_warn_across_families():
    text = render_criterion_scores(
        [_verdict()], "olmoe-1b-7b", "Qwen/Qwen2.5-32B-Instruct"
    )
    assert "WARNING" not in text


# --- Tier 1: pairwise -------------------------------------------------------


def _pair_rows(ticker, criterion, winner_ab, winner_ba):
    base = {
        "ticker": ticker,
        "as_of_a": "2026-07-24T10:00:00",
        "as_of_b": "2026-07-24T14:00:00",
        "model_a": "model-a",
        "model_b": "model-b",
        "judge_model": "judge",
        "criterion": criterion,
        "reason": "because",
    }
    return [
        {**base, "order_shown": "ab", "winner": winner_ab},
        {**base, "order_shown": "ba", "winner": winner_ba},
    ]


def test_pairwise_counts_a_win_only_when_both_orders_agree():
    rows = _pair_rows("AAPL", "news_fidelity", "model_a", "model_a")
    summary = pairwise_summary(rows)
    assert summary["wins_a"] == 1
    assert summary["order_flip_rate"] == 0.0
    assert summary["win_rate_a"] == 1.0


def test_pairwise_treats_an_order_disagreement_as_a_flip_not_a_win():
    """The judge picked whichever record it saw first. That is noise, and
    counting it as a win for either side is how a bake-off invents a result."""
    rows = _pair_rows("AAPL", "news_fidelity", "model_a", "model_b")
    summary = pairwise_summary(rows)
    assert summary["order_flip_rate"] == 1.0
    assert summary["wins_a"] == summary["wins_b"] == 0
    assert summary["win_rate_a"] is None


def test_pairwise_excludes_pairs_unparsable_in_either_order():
    rows = _pair_rows("AAPL", "news_fidelity", "model_a", None)
    summary = pairwise_summary(rows)
    assert summary["unparsed_pairs"] == 1
    assert summary["comparable"] == 0
    assert summary["order_flip_rate"] is None


def test_render_pairwise_refuses_to_call_a_winner_inside_the_noise_floor():
    """A margin smaller than the judge's own order-flip rate is not a result,
    however clean the win/loss table looks."""
    rows = (
        _pair_rows("AAPL", "news_fidelity", "model_a", "model_a")
        + _pair_rows("MSFT", "news_fidelity", "model_b", "model_b")
        + _pair_rows("NVDA", "news_fidelity", "model_a", "model_b")
    )
    text = render_pairwise(rows, "model-a", "model-b")
    assert "no result" in text
    assert "order_flip_rate" in text


def test_render_pairwise_calls_a_winner_outside_the_noise_floor():
    rows = []
    for ticker in ("AAPL", "MSFT", "NVDA", "TSLA"):
        rows += _pair_rows(ticker, "news_fidelity", "model_a", "model_a")
    text = render_pairwise(rows, "model-a", "model-b")
    assert "model-a leads" in text


def test_render_pairwise_of_nothing_does_not_raise():
    assert "not enough" in render_pairwise([], "model-a", "model-b")
