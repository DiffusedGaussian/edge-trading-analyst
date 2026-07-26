"""Tests over the Tier 0 fixture runner.

The whole loop runs against a stubbed chat_completion_full — canned responses,
no server, no network. The last test is the important one: the runner must never
be able to reach yfinance, or a "reproducible offline" fixture run is neither.
"""

from __future__ import annotations

import json
import subprocess
import sys

from eval import run_fixtures
from eval.fixtures import Expectation, Fixture

from edge_analyst import store
from edge_analyst.llm_client import Completion

_GOOD_SENTIMENT = (
    "LABEL: bullish\nSCORE: 7\nCONFIDENCE: high\n"
    "RATIONALE: The new Ohio distribution centre supports demand."
)
_GOOD_DEBATE = (
    "STANCE: Buy\nKEY_POINT: The Ohio centre expands capacity.\nCONFIDENCE: medium"
)
_GOOD_TRADER = (
    "ACTION: Hold\nREASONING: The bull and bear cases offset.\n"
    "ENTRY_PRICE: NA\nSTOP_LOSS: NA\nPOSITION_SIZING: NA"
)

_FIXTURE = Fixture(
    id="probe",
    ticker="TESTA",
    close=100.0,
    rsi=61.7,
    macd_hist=0.4,
    gate_reasons=["macd_bullish_crossover"],
    news_text="TESTA opened a second distribution centre in Ohio.",
    expect=Expectation(
        checks="all",
        allowed_labels=["bullish", "neutral"],
        forbidden_terms=["overbought", "oversold"],
    ),
    note="a probe",
)


def _completion(content: str, finish_reason: str = "stop") -> Completion:
    return Completion(
        content=content,
        finish_reason=finish_reason,
        prompt_tokens=100,
        completion_tokens=40,
        prompt_ms=200.0,
        predicted_ms=800.0,
    )


def _canned(by_stage: dict[str, str], record: list | None = None):
    """Returns a chat_completion_full stand-in that answers per stage, inferred
    from the system prompt (which is what actually differs between stages)."""

    def fake(messages, base_url, model="local", settings=None):
        if record is not None:
            record.append((messages, settings))
        system = messages[0]["content"]
        # Trader first: its prompt mentions "Bull and Bear analysts'", which
        # would otherwise match the bear branch.
        if "Trader/Portfolio Manager" in system:
            key = "trader"
        elif "sentiment analyst" in system:
            key = "sentiment"
        elif "Bull analyst" in system:
            key = "debate_bull"
        else:
            key = "debate_bear"
        return _completion(by_stage[key])

    return fake


_ALL_GOOD = {
    "sentiment": _GOOD_SENTIMENT,
    "debate_bull": _GOOD_DEBATE,
    "debate_bear": _GOOD_DEBATE,
    "trader": _GOOD_TRADER,
}


def _run(tmp_path, **overrides):
    kwargs = {
        "base_url": "http://x",
        "model_name": "test-model",
        "stage": "all",
        "k": 2,
        "seed": 1000,
        "db_path": str(tmp_path / "eval.db"),
        "out_dir": tmp_path / "results",
        "fixtures": [_FIXTURE],
        "completion_fn": _canned(_ALL_GOOD),
        "run_id": "RUN1",
        "verbose": False,
    }
    return run_fixtures.run(**{**kwargs, **overrides})


def test_run_writes_rows_to_both_tables(tmp_path):
    _run(tmp_path)

    conn = store.get_connection(str(tmp_path / "eval.db"))
    run_row = store.fetch_eval_run(conn, "RUN1")
    samples = store.fetch_eval_samples(conn, "RUN1")

    assert run_row["model_name"] == "test-model"
    assert run_row["started_at"] and run_row["finished_at"]
    # 1 fixture x 4 stages x k=2.
    assert len(samples) == 8
    assert {s["stage"] for s in samples} == set(run_fixtures.STAGES)
    assert {s["sample_idx"] for s in samples} == {0, 1}


def test_json_output_matches_the_persisted_rows(tmp_path):
    summary = _run(tmp_path)
    written = json.loads((tmp_path / "results" / "RUN1.json").read_text())

    assert written == json.loads(json.dumps(summary))
    conn = store.get_connection(str(tmp_path / "eval.db"))
    db_rows = store.fetch_eval_samples(conn, "RUN1")
    assert len(db_rows) == len(written["samples"])
    by_key = {(r["fixture_id"], r["stage"], r["sample_idx"]): r for r in db_rows}
    for row in written["samples"]:
        stored = by_key[(row["fixture_id"], row["stage"], row["sample_idx"])]
        assert stored["raw_output"] == row["raw_output"]
        assert json.loads(stored["checks_json"]) == json.loads(row["checks_json"])


def test_synthetic_fixtures_never_touch_the_decisions_table(tmp_path):
    """The boundary the whole two-input-set design rests on."""
    _run(tmp_path)
    conn = store.get_connection(str(tmp_path / "eval.db"))
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM debate_turns").fetchone()[0] == 0


def test_a_well_formed_response_passes_every_declared_check(tmp_path):
    summary = _run(tmp_path)
    for row in summary["samples"]:
        checks = json.loads(row["checks_json"])
        failed = [c for c in checks if not c["passed"]]
        assert not failed, (row["stage"], failed)


def test_the_regression_response_fails_label_consistency(tmp_path):
    """RSI 61.7 is neutral; "strong bullish momentum ... overbought" contradicts
    a label the model was handed. This is the 2026-07-23 gemma-3-1b failure."""
    bad = dict(_ALL_GOOD)
    bad["sentiment"] = (
        "LABEL: bullish\nSCORE: 9\nCONFIDENCE: high\n"
        "RATIONALE: Strong bullish momentum leaves TESTA overbought."
    )
    summary = _run(tmp_path, completion_fn=_canned(bad), stage="sentiment")

    for row in summary["samples"]:
        failed = {c["name"] for c in json.loads(row["checks_json"]) if not c["passed"]}
        assert "label_consistency" in failed


def test_unparseable_response_is_recorded_as_fallbacks_not_as_neutral(tmp_path):
    summary = _run(
        tmp_path,
        stage="sentiment",
        completion_fn=_canned({**_ALL_GOOD, "sentiment": "I cannot help with that."}),
    )
    row = summary["samples"][0]
    assert row["fallbacks"] == "CONFIDENCE,LABEL,RATIONALE,SCORE"
    failed = {c["name"] for c in json.loads(row["checks_json"]) if not c["passed"]}
    assert "fields_parsed" in failed


def test_truncation_is_recorded(tmp_path):
    def truncating(messages, base_url, model="local", settings=None):
        return _completion(_GOOD_SENTIMENT, finish_reason="length")

    summary = _run(tmp_path, stage="sentiment", completion_fn=truncating)
    row = summary["samples"][0]
    assert row["finish_reason"] == "length"
    failed = {c["name"] for c in json.loads(row["checks_json"]) if not c["passed"]}
    assert "not_truncated" in failed


def test_each_sample_gets_its_own_seed(tmp_path):
    calls: list = []
    _run(
        tmp_path,
        stage="sentiment",
        k=3,
        seed=500,
        completion_fn=_canned(_ALL_GOOD, calls),
    )
    assert [settings.seed for _, settings in calls] == [500, 501, 502]
    assert all(settings.temperature == 0.0 for _, settings in calls)


def test_prompts_come_from_the_production_builders(tmp_path):
    calls: list = []
    _run(tmp_path, stage="sentiment", k=1, completion_fn=_canned(_ALL_GOOD, calls))
    messages, _ = calls[0]
    user = messages[1]["content"]
    # format_market_context's exact rendering, including the deterministic label.
    assert "RSI: 61.7 (neutral)" in user
    assert "MACD histogram: 0.400 (bullish momentum)" in user
    assert "distribution centre in Ohio" in user


def test_declared_check_subset_is_honoured_and_excluded_checks_are_absent():
    """A check a fixture excludes must be missing from the record, not recorded
    as a pass — otherwise a scorecard credits a model for a check that never ran."""
    narrow = Fixture(
        **{
            **{k: v for k, v in vars(_FIXTURE).items() if k != "expect"},
            "expect": Expectation(checks=["fields_parsed", "not_truncated"]),
        }
    )
    assert run_fixtures.declared_checks(narrow, "sentiment") == [
        "fields_parsed",
        "not_truncated",
    ]


def test_allowed_label_is_not_applied_to_debate_or_trader_stages():
    """allowed_labels is a sentiment vocabulary; a stance/action is another axis."""
    assert "allowed_label" in run_fixtures.declared_checks(_FIXTURE, "sentiment")
    for stage in ("debate_bull", "debate_bear", "trader"):
        assert "allowed_label" not in run_fixtures.declared_checks(_FIXTURE, stage)


def test_news_checks_are_not_applied_to_the_trader_stage():
    """build_trader_prompt carries no news, so grounding a rationale in it — or
    being hijacked by a sentinel line inside it — is not possible."""
    trader = run_fixtures.declared_checks(_FIXTURE, "trader")
    assert "news_grounding" not in trader
    assert "sentinel_not_hijacked" not in trader
    # But fabrication still counts: it only fires when nothing upstream had news.
    assert "no_fabricated_news" in trader
    # And the debate stages do see the news, so they keep both.
    debate = run_fixtures.declared_checks(_FIXTURE, "debate_bull")
    assert "news_grounding" in debate
    assert "sentinel_not_hijacked" in debate


def test_numeric_fidelity_is_not_applied_to_the_trader_stage():
    """The trader is asked to produce entry/stop levels, so an untraceable
    number in its prose is the job, not a hallucination."""
    assert "numeric_fidelity" not in run_fixtures.declared_checks(_FIXTURE, "trader")
    assert "numeric_fidelity" in run_fixtures.declared_checks(_FIXTURE, "sentiment")


def test_trader_stage_input_state_is_model_independent():
    """The trader prompt must not depend on the debate stage's own output, or a
    bad bull turn shows up as a trader failure."""
    messages = run_fixtures.build_prompt(_FIXTURE, "trader")
    assert "add exposure" in messages[1]["content"]


def test_run_id_is_a_timestamp_plus_model_slug():
    import datetime as dt

    run_id = run_fixtures.make_run_id(
        "OLMoE-1B-7B q4_K_M", dt.datetime(2026, 7, 26, 9, 5)
    )
    assert run_id == "20260726T090500__olmoe-1b-7b-q4-k-m"


def test_runner_does_not_import_data_source():
    """Asserted in a fresh interpreter, not via sys.modules in this one — another
    test could already have imported data_source and masked the regression.

    A fixture run that could reach yfinance would be neither offline nor
    reproducible, and would quietly make synthetic results depend on the market.
    """
    with open(run_fixtures.__file__, encoding="utf-8") as handle:
        import_lines = [
            line
            for line in handle
            if line.startswith(("import ", "from "))
            or line.strip().startswith("import ")
        ]
    assert not [line for line in import_lines if "data_source" in line], import_lines

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import eval.run_fixtures, sys;"
            "print('edge_analyst.data_source' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False", probe.stdout
