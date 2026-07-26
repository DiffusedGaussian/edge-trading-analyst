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
    fallback_rates,
    hard_failures,
    label_flip_rates,
    regressed_checks,
    render_comparison,
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
