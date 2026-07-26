"""Scorecard over one Tier 0 run, and a side-by-side comparison of two.

    python -m eval.report --run <run_id>
    python -m eval.report --compare <run_a> <run_b>
    python -m eval.report --run <run_id> --json

Plain text to stdout, no new dependency. The aggregation functions below are pure
(list of sample rows -> numbers) so they are unit-testable without a run.

Two reporting rules that matter more than they look:

- **A rate with no denominator is `n/a`, never 0.0.** An empty run, a k=1 run's
  standard deviation, and a stage that never ran are all *unknown*, and printing
  0.0 for them reads as a measured result.
- **Prefill and decode throughput are reported separately, each with its own
  spread.** A single blended tok/s number hides the thing you actually care
  about on a Jetson, which is that prompt processing and generation scale
  differently with context length.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass

from edge_analyst import store
from eval.run_fixtures import STAGE_DECISION_FIELD, STAGE_SENTINELS, STAGES


@dataclass(frozen=True)
class Stat:
    """mean +/- population stdev over n observations. `mean is None` means no
    observations at all; `stdev is None` means fewer than two, where spread is
    undefined rather than zero."""

    mean: float | None
    stdev: float | None
    n: int

    def render(self, unit: str = "") -> str:
        if self.mean is None:
            return "n/a"
        spread = "n/a" if self.stdev is None else f"{self.stdev:.1f}"
        return f"{self.mean:.1f} +/- {spread}{unit} (n={self.n})"


def summarise(values: list[float]) -> Stat:
    clean = [v for v in values if v is not None]
    if not clean:
        return Stat(None, None, 0)
    return Stat(
        mean=statistics.fmean(clean),
        # Population stdev: these k samples *are* the run, not a sample of a
        # larger population. Undefined below two observations.
        stdev=statistics.pstdev(clean) if len(clean) > 1 else None,
        n=len(clean),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def render_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def _checks(row: dict) -> list[dict]:
    return json.loads(row["checks_json"] or "[]")


def _parsed(row: dict) -> dict:
    return json.loads(row["parsed_json"] or "{}")


def by_stage(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in samples:
        grouped.setdefault(row["stage"], []).append(row)
    # Fixed stage order, then any unrecognised stage, so reports diff cleanly.
    known = [s for s in STAGES if s in grouped]
    extra = sorted(set(grouped) - set(STAGES))
    return {stage: grouped[stage] for stage in known + extra}


def fallback_rates(samples: list[dict], stage: str) -> list[tuple[str, float | None]]:
    """Per-sentinel fallback rate, worst first. Includes fields that never fell
    back, at 0% — the headline number of the whole harness, so its zeros have to
    be visible rather than merely absent."""
    fields = STAGE_SENTINELS.get(stage) or _observed_fields(samples)
    total = len(samples)
    counts = Counter()
    for row in samples:
        for field in (row["fallbacks"] or "").split(","):
            if field:
                counts[field] += 1
    rates = [(field, _rate(counts.get(field, 0), total)) for field in fields]
    return sorted(rates, key=lambda pair: (-(pair[1] or 0.0), pair[0]))


def _observed_fields(samples: list[dict]) -> list[str]:
    seen: set[str] = set()
    for row in samples:
        seen.update(f for f in (row["fallbacks"] or "").split(",") if f)
    return sorted(seen)


def check_pass_rates(samples: list[dict]) -> list[tuple[str, int, int, float | None]]:
    """(check, passed, ran, rate) worst first. `ran` is per check rather than per
    sample: a check a fixture excluded never ran, and must not be averaged in as
    either a pass or a failure."""
    ran: Counter = Counter()
    passed: Counter = Counter()
    for row in samples:
        for check in _checks(row):
            ran[check["name"]] += 1
            passed[check["name"]] += bool(check["passed"])
    rows = [
        (name, passed[name], ran[name], _rate(passed[name], ran[name])) for name in ran
    ]
    return sorted(rows, key=lambda r: (r[3] if r[3] is not None else 1.0, r[0]))


def truncation_rate(samples: list[dict]) -> float | None:
    truncated = sum(1 for row in samples if row["finish_reason"] == "length")
    return _rate(truncated, len(samples))


def label_flip_rates(
    samples: list[dict], stage: str
) -> list[tuple[str, float | None, str]]:
    """Per fixture, the fraction of samples disagreeing with the modal decision.

    Instability at a fixed temperature and seed is itself a model-quality
    signal — and on a bake-off it bounds how much of any scorecard difference
    could be noise rather than capability.
    """
    field = STAGE_DECISION_FIELD.get(stage)
    if field is None:
        return []
    by_fixture: dict[str, list[str]] = {}
    for row in samples:
        value = str(_parsed(row).get(field, ""))
        by_fixture.setdefault(row["fixture_id"], []).append(value)

    rows = []
    for fixture_id, values in sorted(by_fixture.items()):
        counts = Counter(values)
        modal, modal_count = counts.most_common(1)[0]
        rows.append((fixture_id, _rate(len(values) - modal_count, len(values)), modal))
    return sorted(rows, key=lambda r: (-(r[1] or 0.0), r[0]))


def throughput(samples: list[dict]) -> tuple[Stat, Stat]:
    """(prefill tok/s, decode tok/s). Derived here rather than in llm_client,
    which stays a transport. Rows without llama.cpp `timings` contribute
    nothing rather than a zero."""

    def rates(token_key: str, ms_key: str) -> list[float]:
        values = []
        for row in samples:
            tokens, ms = row.get(token_key), row.get(ms_key)
            if tokens and ms and ms > 0:
                values.append(tokens / (ms / 1000.0))
        return values

    return (
        summarise(rates("prompt_tokens", "prompt_ms")),
        summarise(rates("completion_tokens", "predicted_ms")),
    )


def hard_failures(samples: list[dict]) -> list[tuple[str, str, str, str]]:
    """Every (fixture, stage, check, detail) that failed, deduplicated across the
    k samples — the same failure on 5 samples is one problem, not five."""
    seen: dict[tuple[str, str, str], str] = {}
    for row in samples:
        for check in _checks(row):
            if not check["passed"]:
                key = (row["fixture_id"], row["stage"], check["name"])
                seen.setdefault(key, check["detail"])
    return [(f, s, c, detail) for (f, s, c), detail in sorted(seen.items())]


def build_report(run: dict | None, samples: list[dict]) -> dict:
    """The whole scorecard as plain data, so --json and the text renderer can't
    drift apart."""
    stages = by_stage(samples)
    return {
        "run": run or {},
        "sample_count": len(samples),
        "fixture_count": len({row["fixture_id"] for row in samples}),
        "stages": {
            stage: {
                "sample_count": len(rows),
                "fallback_rates": [
                    {"field": field, "rate": rate}
                    for field, rate in fallback_rates(rows, stage)
                ],
                "check_pass_rates": [
                    {"check": name, "passed": passed, "ran": ran, "rate": rate}
                    for name, passed, ran, rate in check_pass_rates(rows)
                ],
                "truncation_rate": truncation_rate(rows),
                "label_flip_rates": [
                    {"fixture_id": fixture, "rate": rate, "modal": modal}
                    for fixture, rate, modal in label_flip_rates(rows, stage)
                ],
                "prefill_tok_s": _stat_dict(throughput(rows)[0]),
                "decode_tok_s": _stat_dict(throughput(rows)[1]),
            }
            for stage, rows in stages.items()
        },
        "hard_failures": [
            {"fixture_id": f, "stage": s, "check": c, "detail": d}
            for f, s, c, d in hard_failures(samples)
        ],
    }


def _stat_dict(stat: Stat) -> dict:
    return {"mean": stat.mean, "stdev": stat.stdev, "n": stat.n}


# --- rendering ---------------------------------------------------------------


def render_run(report: dict) -> str:
    run = report["run"]
    lines = [
        f"run       {run.get('run_id', '?')}",
        f"model     {run.get('model_name', '?')}",
        f"settings  stage={run.get('stage')} k={run.get('k')} "
        f"seed={run.get('seed')} temperature={run.get('temperature')}",
        f"code      git {run.get('git_sha') or 'unknown'}",
        f"samples   {report['sample_count']} over {report['fixture_count']} fixture(s)",
    ]
    if not report["sample_count"]:
        lines.append("\nno samples recorded for this run")
        return "\n".join(lines)

    for stage, data in report["stages"].items():
        lines.append(f"\n=== {stage} ({data['sample_count']} samples) ===")

        lines.append("\n  fallback rate per sentinel field (worst first)")
        for entry in data["fallback_rates"]:
            lines.append(f"    {entry['field']:<18}{render_rate(entry['rate'])}")

        lines.append("\n  check pass rate (worst first)")
        for entry in data["check_pass_rates"]:
            lines.append(
                f"    {entry['check']:<22}{render_rate(entry['rate']):>5}"
                f"   ({entry['passed']}/{entry['ran']})"
            )

        lines.append(f"\n  truncation rate     {render_rate(data['truncation_rate'])}")

        flips = [e for e in data["label_flip_rates"] if (e["rate"] or 0) > 0]
        if flips:
            lines.append("  label-flip rate (unstable fixtures)")
            for entry in flips:
                lines.append(
                    f"    {entry['fixture_id']:<30}{render_rate(entry['rate'])}"
                    f"  (modal={entry['modal']})"
                )
        elif data["label_flip_rates"]:
            lines.append("  label-flip rate     0% (every fixture stable)")

        lines.append(
            f"  prefill  {_render_stat(data['prefill_tok_s'], ' tok/s')}\n"
            f"  decode   {_render_stat(data['decode_tok_s'], ' tok/s')}"
        )

    if report["hard_failures"]:
        lines.append(f"\n=== hard failures ({len(report['hard_failures'])}) ===")
        for entry in report["hard_failures"]:
            lines.append(
                f"  {entry['fixture_id']} / {entry['stage']} / {entry['check']}\n"
                f"      {entry['detail']}"
            )
    else:
        lines.append("\nno hard failures")
    return "\n".join(lines)


def _render_stat(data: dict, unit: str) -> str:
    return Stat(data["mean"], data["stdev"], data["n"]).render(unit)


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{(b - a) * 100:+.0f}pp"


def render_comparison(report_a: dict, report_b: dict) -> str:
    name_a = report_a["run"].get("model_name", "A")
    name_b = report_b["run"].get("model_name", "B")
    lines = [
        f"A  {report_a['run'].get('run_id', '?')}  {name_a}",
        f"B  {report_b['run'].get('run_id', '?')}  {name_b}",
    ]
    if report_a["run"].get("k") != report_b["run"].get("k") or report_a["run"].get(
        "temperature"
    ) != report_b["run"].get("temperature"):
        lines.append(
            "  WARNING: the two runs used different k/temperature — the "
            "difference below mixes model quality with sampling settings"
        )

    for stage in sorted(set(report_a["stages"]) | set(report_b["stages"])):
        stage_a = report_a["stages"].get(stage, {})
        stage_b = report_b["stages"].get(stage, {})
        lines.append(f"\n=== {stage} ===")
        lines.append(f"  {'check':<24}{'A':>6}{'B':>8}{'delta':>10}")
        rates_a = {e["check"]: e["rate"] for e in stage_a.get("check_pass_rates", [])}
        rates_b = {e["check"]: e["rate"] for e in stage_b.get("check_pass_rates", [])}
        for check in sorted(set(rates_a) | set(rates_b)):
            lines.append(
                f"  {check:<24}{render_rate(rates_a.get(check)):>6}"
                f"{render_rate(rates_b.get(check)):>8}"
                f"{_delta(rates_a.get(check), rates_b.get(check)):>10}"
            )

        lines.append(f"  {'field fallback':<24}{'A':>6}{'B':>8}{'delta':>10}")
        fb_a = {e["field"]: e["rate"] for e in stage_a.get("fallback_rates", [])}
        fb_b = {e["field"]: e["rate"] for e in stage_b.get("fallback_rates", [])}
        for field in sorted(set(fb_a) | set(fb_b)):
            lines.append(
                f"  {field:<24}{render_rate(fb_a.get(field)):>6}"
                f"{render_rate(fb_b.get(field)):>8}"
                f"{_delta(fb_a.get(field), fb_b.get(field)):>10}"
            )

        for label, key in (("prefill", "prefill_tok_s"), ("decode ", "decode_tok_s")):
            side_a = _render_stat(stage_a.get(key, _EMPTY), " tok/s")
            side_b = _render_stat(stage_b.get(key, _EMPTY), " tok/s")
            lines.append(f"  {label}  A {side_a}   B {side_b}")

    regressions = regressed_checks(report_a, report_b)
    if regressions:
        lines.append(
            f"\n=== REGRESSED: passed in A, failed in B ({len(regressions)}) ==="
        )
        for fixture, stage, check in regressions:
            lines.append(f"  {fixture} / {stage} / {check}")
    else:
        lines.append("\nno check regressed from pass to fail")

    fixed = regressed_checks(report_b, report_a)
    if fixed:
        lines.append(f"\nfixed: failed in A, passes in B ({len(fixed)})")
        for fixture, stage, check in fixed:
            lines.append(f"  {fixture} / {stage} / {check}")
    return "\n".join(lines)


_EMPTY = {"mean": None, "stdev": None, "n": 0}


def regressed_checks(report_a: dict, report_b: dict) -> list[tuple[str, str, str]]:
    """(fixture, stage, check) that passed everywhere in A and failed at least
    once in B. Restricted to pairs B actually ran, so a stage or check absent
    from B is not reported as a regression."""
    failed_a = {
        (e["fixture_id"], e["stage"], e["check"]) for e in report_a["hard_failures"]
    }
    failed_b = {
        (e["fixture_id"], e["stage"], e["check"]) for e in report_b["hard_failures"]
    }
    ran_a = _ran_pairs(report_a)
    return sorted(pair for pair in failed_b - failed_a if pair in ran_a)


def _ran_pairs(report: dict) -> set[tuple[str, str, str]]:
    """Which (fixture, stage, check) triples a report actually covers. Derived
    from the run's fixtures x the checks each stage ran, so "absent" and "passed"
    stay distinguishable."""
    pairs = set()
    for stage, data in report["stages"].items():
        checks = [e["check"] for e in data["check_pass_rates"]]
        fixtures = [e["fixture_id"] for e in data["label_flip_rates"]]
        for fixture in fixtures:
            for check in checks:
                pairs.add((fixture, stage, check))
    return pairs


# --- Tier 1: judge reporting -------------------------------------------------


def criterion_scores(verdicts: list[dict]) -> dict[str, dict]:
    """Per criterion: derived_score and parse_failure_rate, reported separately
    and never blended.

    derived_score is yes/answered, replacing the judge's own OVERALL_SCORE — an
    LLM's absolute 0-10 clusters at 7-8 and carries almost no resolution. The
    parse-failure rate sits beside it because a run where the judge often failed
    to answer is not a low-scoring run, it is an invalid one, and only the two
    numbers together distinguish those.
    """
    by_criterion: dict[str, list[dict]] = {}
    for row in verdicts:
        by_criterion.setdefault(row["criterion"], []).append(row)

    scores = {}
    for criterion, rows in sorted(by_criterion.items()):
        answered = [r for r in rows if r["verdict"] is not None]
        yes = sum(1 for r in answered if r["verdict"] == "yes")
        scores[criterion] = {
            "derived_score": _rate(yes, len(answered)),
            "yes": yes,
            "answered": len(answered),
            "judged": len(rows),
            "parse_failure_rate": _rate(len(rows) - len(answered), len(rows)),
        }
    return scores


def render_criterion_scores(
    verdicts: list[dict], model: str = "", judge_model: str = ""
) -> str:
    from eval.rubric import same_family

    lines = [f"judge {judge_model or '?'}   model {model or '(mixed)'}"]
    if model and judge_model and same_family(model, judge_model):
        lines.append(
            f"  WARNING: {model} and {judge_model} look like the same model "
            "family — these verdicts are self-preference-prone; re-run with "
            "--second-judge from another family before trusting them"
        )
    scores = criterion_scores(verdicts)
    if not scores:
        return "\n".join(lines + ["  no verdicts recorded"])

    lines.append(f"  {'criterion':<24}{'derived':>8}{'yes/ans':>10}{'parse-fail':>12}")
    for criterion, data in scores.items():
        derived = (
            "n/a" if data["derived_score"] is None else f"{data['derived_score']:.2f}"
        )
        lines.append(
            f"  {criterion:<24}{derived:>8}"
            f"{f'{data["yes"]}/{data["answered"]}':>10}"
            f"{render_rate(data['parse_failure_rate']):>12}"
        )
    worst = max((d["parse_failure_rate"] or 0.0) for d in scores.values())
    if worst >= 0.2:
        lines.append(
            f"  a criterion failed to parse {worst:.0%} of the time — treat this "
            "run as invalid, not as a low score, and fix CRITERIA[...] first"
        )
    return "\n".join(lines)


def pairwise_summary(rows: list[dict]) -> dict:
    """Win rates after de-duplicating display order, plus the order_flip_rate.

    order_flip_rate is the headline: it is the judge's measured noise floor. Any
    win-rate gap smaller than it is not a result, however clean the table looks.
    """
    by_pair: dict[tuple, dict[str, str | None]] = {}
    for row in rows:
        key = (row["ticker"], row["as_of_a"], row["as_of_b"], row["criterion"])
        by_pair.setdefault(key, {})[row["order_shown"]] = row["winner"]

    wins_a = wins_b = ties = flips = comparable = unparsed = 0
    for orders in by_pair.values():
        winners = [orders.get(order) for order in ("ab", "ba")]
        if any(w is None for w in winners):
            unparsed += 1
            continue
        comparable += 1
        if winners[0] != winners[1]:
            flips += 1
            continue
        # Both orders agree, so this pair has a real winner.
        if winners[0] == "model_a":
            wins_a += 1
        elif winners[0] == "model_b":
            wins_b += 1
        else:
            ties += 1

    decided = wins_a + wins_b
    return {
        "pairs": len(by_pair),
        "comparable": comparable,
        "unparsed_pairs": unparsed,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": _rate(wins_a, decided),
        "order_flip_rate": _rate(flips, comparable),
    }


def render_pairwise(rows: list[dict], model_a: str, model_b: str) -> str:
    summary = pairwise_summary(rows)
    flip = summary["order_flip_rate"]
    win_rate = summary["win_rate_a"]
    lines = [
        f"\nA = {model_a}   B = {model_b}",
        f"  pairs judged        {summary['pairs']} "
        f"({summary['unparsed_pairs']} unparsable in at least one order)",
        f"  A wins              {summary['wins_a']}",
        f"  B wins              {summary['wins_b']}",
        f"  ties                {summary['ties']}",
        f"  win_rate_a          {render_rate(win_rate)} "
        "(order-consistent decisions only)",
        f"  order_flip_rate     {render_rate(flip)}   <-- the judge's noise floor",
    ]
    if win_rate is None or flip is None:
        lines.append(
            "  not enough order-consistent verdicts to call a winner either way"
        )
    else:
        margin = abs(win_rate - 0.5) * 2
        if margin <= flip:
            lines.append(
                f"  VERDICT: no result. The A-vs-B margin ({margin:.0%}) is within "
                f"the judge's own order-flip noise ({flip:.0%})."
            )
        else:
            leader = model_a if win_rate > 0.5 else model_b
            lines.append(
                f"  VERDICT: {leader} leads by {margin:.0%}, outside the "
                f"{flip:.0%} order-flip noise floor."
            )
    return "\n".join(lines)


# --- entrypoint --------------------------------------------------------------


def load(db_path: str, run_id: str) -> dict:
    conn = store.get_connection(db_path)
    try:
        return build_report(
            store.fetch_eval_run(conn, run_id), store.fetch_eval_samples(conn, run_id)
        )
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tier 0 scorecard and comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="run_id to summarise ('latest' for the newest)")
    group.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    group.add_argument(
        "--judge",
        action="store_true",
        help="Tier 1: per-criterion derived scores from criterion_verdicts",
    )
    group.add_argument(
        "--pairwise",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Tier 1: pairwise win rate and order-flip noise floor",
    )
    parser.add_argument("--db", default="data/edge_analyst.db")
    parser.add_argument("--model", default="", help="filter Tier 1 output to a model")
    parser.add_argument("--judge-model", default="", help="filter to one judge")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.judge:
        conn = store.get_connection(args.db)
        verdicts = store.fetch_criterion_verdicts(
            conn, judge_model=args.judge_model or None, model=args.model or None
        )
        conn.close()
        if args.json:
            print(json.dumps(criterion_scores(verdicts), indent=2))
        else:
            print(render_criterion_scores(verdicts, args.model, args.judge_model))
        return

    if args.pairwise:
        model_a, model_b = args.pairwise
        conn = store.get_connection(args.db)
        rows = store.fetch_pairwise_results(
            conn, model_a, model_b, args.judge_model or None
        )
        conn.close()
        if args.json:
            print(json.dumps(pairwise_summary(rows), indent=2))
        else:
            print(render_pairwise(rows, model_a, model_b))
        return

    if args.run:
        run_id = args.run
        if run_id == "latest":
            conn = store.get_connection(args.db)
            run_id = store.latest_eval_run_id(conn) or ""
            conn.close()
            if not run_id:
                print("no eval runs recorded")
                return
        report = load(args.db, run_id)
        print(json.dumps(report, indent=2) if args.json else render_run(report))
        return

    report_a, report_b = (load(args.db, run_id) for run_id in args.compare)
    if args.json:
        print(
            json.dumps(
                {
                    "a": report_a,
                    "b": report_b,
                    "regressed": regressed_checks(report_a, report_b),
                },
                indent=2,
            )
        )
    else:
        print(render_comparison(report_a, report_b))


if __name__ == "__main__":
    main()
