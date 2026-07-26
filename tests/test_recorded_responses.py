"""Replay real captured model output through the parsers and Tier 0 checks.

Every other parser test in this suite uses hand-written strings, which encode
what we *imagine* a small model emits. This module replays what one actually
emitted, so parser robustness becomes a regression gate against reality. It runs
on ubuntu-latest with no GPU and no model, so it belongs in the existing CI
`quality` job with no workflow change.

Populate tests/fixtures/responses/ from a real device run:

    uv run python -m eval.run_fixtures --base-url http://localhost:8080 \\
        --model-name gemma-3-1b-it --k 1 \\
        --export-fixtures tests/fixtures/responses/

Until then the parametrised tests skip, and the skip is reported in the pytest
summary (`addopts = -ra`) rather than passing silently — an empty replay suite
must not read as a covered one.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from eval import run_fixtures
from eval.fixtures import load_synthetic

from edge_analyst.llm_client import Completion

RESPONSES_DIR = Path(__file__).parent / "fixtures" / "responses"
SIDECAR = RESPONSES_DIR / "expected.yaml"


def _load_expected() -> dict:
    if not SIDECAR.exists():
        return {}
    with SIDECAR.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


_EXPECTED = _load_expected()
_CASES = sorted(_EXPECTED)

_NO_CAPTURES = (
    "no captured responses yet — populate tests/fixtures/responses/ with "
    "`python -m eval.run_fixtures --export-fixtures tests/fixtures/responses/` "
    "against a real llama-server (see this module's docstring)"
)


def test_capture_directory_is_wired_up():
    """Fails if the directory itself went missing, so the replay suite can't
    quietly disappear. Skips (visibly) while it is merely empty."""
    assert RESPONSES_DIR.is_dir(), f"{RESPONSES_DIR} is missing"
    if not _CASES:
        pytest.skip(_NO_CAPTURES)


@pytest.mark.parametrize("name", _CASES)
def test_recorded_response_replays_identically(name: str):
    if not _CASES:
        pytest.skip(_NO_CAPTURES)
    expected = _EXPECTED[name]
    raw = (RESPONSES_DIR / name).read_text(encoding="utf-8")

    fixtures = {f.id: f for f in load_synthetic()}
    fixture = fixtures.get(expected["fixture_id"])
    assert fixture is not None, (
        f"{name} references fixture {expected['fixture_id']!r}, which no longer "
        "exists — delete the capture or restore the fixture"
    )

    stage = expected["stage"]
    completion = Completion(
        content=raw,
        finish_reason=expected["finish_reason"],
        prompt_tokens=None,
        completion_tokens=None,
        prompt_ms=None,
        predicted_ms=None,
    )
    result = run_fixtures.score_sample(
        fixture,
        stage,
        0,
        run_fixtures.build_prompt(fixture, stage),
        completion,
    )

    assert sorted(result.fallbacks) == list(expected["fallbacks"]), (
        f"{name}: parser fallbacks changed"
    )
    actual_checks = {check.name: check.passed for check in result.checks}
    assert actual_checks == dict(expected["checks"]), (
        f"{name}: check outcomes changed — "
        f"{[asdict(c) for c in result.checks if not c.passed]}"
    )


def test_at_least_one_capture_is_malformed():
    """The valuable captures are the broken ones. A directory of only
    well-formed responses proves the parser handles the easy case, which the
    hand-written tests already cover."""
    if not _CASES:
        pytest.skip(_NO_CAPTURES)
    assert any(
        _EXPECTED[name]["fallbacks"] or not all(_EXPECTED[name]["checks"].values())
        for name in _CASES
    ), (
        "every captured response parses cleanly and passes every check — capture "
        "a genuinely malformed one, those are the regression cases that matter"
    )
