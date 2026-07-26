"""Tier 0 runner: every synthetic fixture x k samples through one model, scored
by eval/checks.py alone. No judge, no GPU, no network beyond the llama-server
endpoint under test.

    python -m eval.run_fixtures \
        --base-url http://localhost:8081 \
        --model-name olmoe-1b-7b-q4km \
        --k 5 --seed 1234 --stage all

Two deliberate constraints:

- **Prompts come from production.** build_sentiment_prompt / build_debate_prompt
  / build_trader_prompt are imported, never reimplemented. A parallel prompt
  builder in eval drifts from the real one and then measures nothing.
- **This module must never import edge_analyst.data_source.** Fixture inputs are
  hand-built; a fixture run that reached yfinance would be neither reproducible
  nor offline, and there is a test asserting the import is absent.

Results land in `eval_runs` / `eval_samples` — never in `decisions`, which holds
real market history. A check a fixture excludes (see its `expect.checks`) is
absent from the record rather than recorded as passing, so a scorecard never
credits a model for a check that didn't run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from edge_analyst import store
from edge_analyst.debate import (
    DebateState,
    DebateTurn,
    build_debate_prompt,
    build_trader_prompt,
    parse_debate_response,
    parse_trader_response,
)
from edge_analyst.indicators import interpret_macd_hist, interpret_rsi
from edge_analyst.llm_client import Completion, GenSettings, chat_completion_full
from edge_analyst.news_analyst import build_sentiment_prompt, parse_sentiment_response
from eval.checks import CHECK_ORDER, CheckInputs, CheckResult, run_all_checks
from eval.fixtures import Fixture, load_synthetic

STAGES = ("sentiment", "debate_bull", "debate_bear", "trader")

# The sentinel fields each stage is supposed to emit, and the parsed field that
# carries its decision. Shared with eval/report.py so a fallback rate can be
# reported per field including the fields that never fell back — a rate derived
# only from observed failures silently omits the 0% rows.
STAGE_SENTINELS: dict[str, tuple[str, ...]] = {
    "sentiment": ("LABEL", "SCORE", "CONFIDENCE", "RATIONALE"),
    "debate_bull": ("STANCE", "KEY_POINT", "CONFIDENCE"),
    "debate_bear": ("STANCE", "KEY_POINT", "CONFIDENCE"),
    "trader": ("ACTION", "REASONING", "ENTRY_PRICE", "STOP_LOSS", "POSITION_SIZING"),
}

STAGE_DECISION_FIELD: dict[str, str] = {
    "sentiment": "label",
    "debate_bull": "stance",
    "debate_bear": "stance",
    "trader": "action",
}

# The stage names --stage accepts, and which concrete stages each expands to.
STAGE_GROUPS = {
    "sentiment": ("sentiment",),
    "debate": ("debate_bull", "debate_bear"),
    "trader": ("trader",),
    "all": STAGES,
}

# Not every check is applicable to every stage, and applying one that isn't
# manufactures failures on correct output. Excluded rather than trivially passed,
# so a scorecard never credits a model for a check that couldn't have run.
#
# - `allowed_label` is a sentiment vocabulary (bullish/bearish/neutral); a debate
#   stance or a trader action is a different axis entirely.
# - build_trader_prompt carries no news — only the indicator block and the
#   bull/bear positions. Grounding a rationale in news the stage never saw is
#   impossible, and an injected sentinel line in the news cannot reach it.
#   (no_fabricated_news is deliberately kept: it fires only when the fixture has
#   no news at all, in which case nothing upstream had news either.)
# - the trader is *asked* to produce price levels, so a number in its prose that
#   isn't traceable to the input is expected behaviour, not fabrication.
_STAGE_EXCLUDED_CHECKS = {
    "debate_bull": frozenset({"allowed_label"}),
    "debate_bear": frozenset({"allowed_label"}),
    "trader": frozenset(
        {
            "allowed_label",
            "news_grounding",
            "sentinel_not_hijacked",
            "numeric_fidelity",
        }
    ),
}

# A fixed, model-independent standoff handed to the trader stage. Deriving it
# from the debate stage's own output instead would mean a bad bull turn showing
# up as a trader failure — the stages have to be independently attributable.
# Deliberately free of indicator vocabulary so it can't leak a label into the
# prompt that check_label_consistency would then read back out.
_TRADER_INPUT_STATE = DebateState(
    round=1,
    bull=DebateTurn(
        stance="buy",
        key_point="The news gives a concrete reason to add exposure here.",
        confidence="medium",
    ),
    bear=DebateTurn(
        stance="sell",
        key_point="The setup can reverse quickly and the news is not decisive.",
        confidence="medium",
    ),
)


@dataclass(frozen=True)
class SampleResult:
    fixture_id: str
    stage: str
    sample_idx: int
    raw_output: str
    parsed: dict
    fallbacks: frozenset[str]
    completion: Completion
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def make_run_id(model_name: str, now: dt.datetime | None = None) -> str:
    stamp = (now or dt.datetime.now()).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}__{_slug(model_name)}"


def git_sha() -> str | None:
    """Short HEAD sha, so a committed scorecard is traceable to the code that
    produced it. None outside a git checkout rather than an error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def build_prompt(fixture: Fixture, stage: str) -> list[dict]:
    """Delegates to the production prompt builders — the point of the runner is
    to measure what the real cascade sends, not an eval-local approximation."""
    args = (
        fixture.ticker,
        fixture.close,
        fixture.rsi,
        fixture.macd_hist,
        fixture.gate_reasons,
    )
    if stage == "sentiment":
        return build_sentiment_prompt(*args, fixture.news_text)
    if stage in ("debate_bull", "debate_bear"):
        persona = "bull" if stage == "debate_bull" else "bear"
        # No opposing key point: round 1. The round-2 rebuttal dynamic depends
        # on the model's own round-1 output, which makes it a cascade property
        # rather than a single-response one — Tier 1 territory, not Tier 0.
        return build_debate_prompt(persona, *args, fixture.news_text, None)
    if stage == "trader":
        return build_trader_prompt(*args, _TRADER_INPUT_STATE)
    raise ValueError(f"unknown stage: {stage}")


def _parse_for_stage(
    stage: str, text: str
) -> tuple[dict, frozenset[str], str, str, dict]:
    """Returns (parsed_dict, fallbacks, free_text, label, parsed_values).

    `free_text` is the model's prose only — the parsed RATIONALE / KEY_POINT /
    REASONING value. Never the raw response: some models echo the prompt's
    indicator block, whose labels are ours, not the model's claim.
    """
    if stage == "sentiment":
        signal = parse_sentiment_response(text)
        return (
            asdict(signal) | {"fallbacks": sorted(signal.fallbacks)},
            signal.fallbacks,
            signal.rationale,
            signal.label,
            dict(
                zip(
                    STAGE_SENTINELS["sentiment"],
                    (
                        signal.label,
                        _format_number(signal.score),
                        signal.confidence,
                        signal.rationale,
                    ),
                    strict=True,
                )
            ),
        )
    if stage in ("debate_bull", "debate_bear"):
        turn = parse_debate_response(text)
        return (
            asdict(turn) | {"fallbacks": sorted(turn.fallbacks)},
            turn.fallbacks,
            turn.key_point,
            "",
            {
                "STANCE": turn.stance,
                "KEY_POINT": turn.key_point,
                "CONFIDENCE": turn.confidence,
            },
        )
    if stage == "trader":
        decision = parse_trader_response(text)
        return (
            asdict(decision) | {"fallbacks": sorted(decision.fallbacks)},
            decision.fallbacks,
            decision.reasoning,
            "",
            {
                "ACTION": decision.action,
                "REASONING": decision.reasoning,
                "ENTRY_PRICE": _format_number(decision.entry_price),
                "STOP_LOSS": _format_number(decision.stop_loss),
                "POSITION_SIZING": _format_number(decision.position_sizing),
            },
        )
    raise ValueError(f"unknown stage: {stage}")


def _format_number(value: float | None) -> str:
    """Renders a parsed number back the way a model would have written it, so
    check_sentinel_not_hijacked can compare it against an injected literal."""
    if value is None:
        return "NA"
    return str(int(value)) if float(value).is_integer() else str(value)


def declared_checks(fixture: Fixture, stage: str) -> list[str]:
    """The checks this fixture asks for at this stage, in CHECK_ORDER."""
    requested = (
        set(CHECK_ORDER)
        if fixture.expect.checks == "all"
        else set(fixture.expect.checks)
    )
    requested -= _STAGE_EXCLUDED_CHECKS.get(stage, frozenset())
    return [name for name in CHECK_ORDER if name in requested]


def score_sample(
    fixture: Fixture,
    stage: str,
    sample_idx: int,
    messages: list[dict],
    completion: Completion,
) -> SampleResult:
    """Everything after the HTTP call: parse, check, filter to what the fixture
    declared. Pure, so tests can drive it with canned model output."""
    parsed, fallbacks, free_text, label, parsed_values = _parse_for_stage(
        stage, completion.content
    )
    results = run_all_checks(
        CheckInputs(
            raw_output=completion.content,
            prompt_text="\n".join(m["content"] for m in messages),
            prompt_boilerplate=messages[0]["content"],
            free_text=free_text,
            rsi_label=interpret_rsi(fixture.rsi),
            macd_label=interpret_macd_hist(fixture.macd_hist),
            news_text=fixture.news_text,
            fallbacks=fallbacks,
            finish_reason=completion.finish_reason,
            parsed_values=parsed_values,
            label=label,
            allowed_labels=fixture.expect.allowed_labels,
            forbidden_terms=fixture.expect.forbidden_terms,
        )
    )
    wanted = set(declared_checks(fixture, stage))
    return SampleResult(
        fixture_id=fixture.id,
        stage=stage,
        sample_idx=sample_idx,
        raw_output=completion.content,
        parsed=parsed,
        fallbacks=fallbacks,
        completion=completion,
        checks=[r for r in results if r.name in wanted],
    )


def sample_to_row(run_id: str, result: SampleResult) -> dict:
    return {
        "run_id": run_id,
        "fixture_id": result.fixture_id,
        "sample_idx": result.sample_idx,
        "stage": result.stage,
        "raw_output": result.raw_output,
        "parsed_json": json.dumps(result.parsed, sort_keys=True),
        "fallbacks": ",".join(sorted(result.fallbacks)),
        "finish_reason": result.completion.finish_reason,
        "prompt_ms": result.completion.prompt_ms,
        "predicted_ms": result.completion.predicted_ms,
        "prompt_tokens": result.completion.prompt_tokens,
        "completion_tokens": result.completion.completion_tokens,
        "checks_json": json.dumps([asdict(c) for c in result.checks]),
    }


def run(
    *,
    base_url: str,
    model_name: str,
    stage: str = "all",
    k: int = 5,
    seed: int = 1234,
    temperature: float = 0.0,
    max_tokens: int = 512,
    db_path: str = "data/edge_analyst.db",
    out_dir: str | Path = "eval/results",
    fixtures: list[Fixture] | None = None,
    completion_fn=chat_completion_full,
    run_id: str | None = None,
    verbose: bool = True,
) -> dict:
    """Runs the matrix and returns the run's JSON-shaped summary.

    `completion_fn` is injected so tests can drive the whole loop with canned
    responses and no server.
    """
    fixtures = fixtures if fixtures is not None else load_synthetic()
    stages = STAGE_GROUPS[stage]
    run_id = run_id or make_run_id(model_name)
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    conn = store.get_connection(db_path)
    run_row = {
        "run_id": run_id,
        "model_name": model_name,
        "base_url": base_url,
        "stage": stage,
        "k": k,
        "seed": seed,
        "temperature": temperature,
        "started_at": started_at,
        "finished_at": None,
        "git_sha": git_sha(),
    }
    # Written up front so a run that dies mid-way still leaves a record of what
    # was attempted, rather than looking like it never happened.
    store.save_eval_run(conn, run_row)

    rows = []
    try:
        for fixture in fixtures:
            for fixture_stage in stages:
                for sample_idx in range(k):
                    messages = build_prompt(fixture, fixture_stage)
                    settings = GenSettings(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        # Distinct per sample: at temperature 0 the k samples are
                        # only meaningfully different if the server honours the
                        # seed. Run again at --temperature 0.7 for a real
                        # self-consistency signal.
                        seed=seed + sample_idx,
                    )
                    completion = completion_fn(messages, base_url, settings=settings)
                    result = score_sample(
                        fixture, fixture_stage, sample_idx, messages, completion
                    )
                    row = sample_to_row(run_id, result)
                    store.save_eval_sample(conn, row)
                    rows.append(row)
                    if verbose:
                        _print_sample(result)
    finally:
        run_row["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        store.save_eval_run(conn, run_row)
        conn.close()

    summary = {"run": run_row, "samples": rows}
    out_path = Path(out_dir) / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if verbose:
        _print_summary(rows, out_path)
    return summary


def _print_sample(result: SampleResult) -> None:
    """One line per sample, printed as it lands — a 16-fixture x 5-sample run on
    a Jetson takes long enough that silence is indistinguishable from a hang."""
    failed = [c.name for c in result.checks if not c.passed]
    verdict = "PASS" if not failed else f"FAIL ({', '.join(failed)})"
    print(
        f"  {result.fixture_id:<30} {result.stage:<12} #{result.sample_idx} {verdict}"
    )


def _print_summary(rows: list[dict], out_path: Path) -> None:
    total = len(rows)
    clean = sum(
        1 for row in rows if all(c["passed"] for c in json.loads(row["checks_json"]))
    )
    print(
        f"\n{clean}/{total} samples passed every declared check "
        f"({total - clean} with at least one failure)"
    )
    print(f"wrote {out_path}")
    print("run `python -m eval.report --run <run_id>` for the scorecard")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tier 0 synthetic fixture runner")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument(
        "--model-name",
        required=True,
        help="label recorded on the run, e.g. olmoe-1b-7b-q4km",
    )
    parser.add_argument("--stage", default="all", choices=sorted(STAGE_GROUPS))
    parser.add_argument("--k", type=int, default=5, help="samples per fixture")
    parser.add_argument(
        "--seed", type=int, default=1234, help="base seed; +i per sample"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 for reproducibility; run again at 0.7 for self-consistency",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--db", default="data/edge_analyst.db")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args(argv)

    print(
        f"tier 0: {args.model_name} @ {args.base_url} (stage={args.stage}, k={args.k})"
    )
    run(
        base_url=args.base_url,
        model_name=args.model_name,
        stage=args.stage,
        k=args.k,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        db_path=args.db,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
