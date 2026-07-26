"""Loader for the synthetic fixture set (eval/fixtures/synthetic/*.yaml).

Synthetic fixtures are hand-built market situations with a known-correct answer,
so Tier 0's checks become hard pass/fail regression gates. They are kept
strictly separate from replayed real days (the `decisions` table), and the two
sets' metrics are never averaged: a synthetic score measures "does the model
handle this specific adversarial situation", a replayed score measures "which
model is better on real data", and mixing them answers neither question.

Every ticker is synthetic (see SYNTHETIC_TICKERS) so a fixture run can never be
mistaken for real market data, and the runner never writes fixtures to
`decisions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from eval.checks import CHECK_ORDER

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"

# Deliberately not real tickers. If one of these ever shows up in the decisions
# table or a P&L report, something has crossed the synthetic/real boundary.
SYNTHETIC_TICKERS = frozenset({"TESTA", "TESTB", "TESTC", "TESTD"})

_REQUIRED_KEYS = frozenset(
    {"id", "ticker", "close", "rsi", "macd_hist", "gate_reasons", "news_text", "expect"}
)
_OPTIONAL_KEYS = frozenset({"note"})
_EXPECT_KEYS = frozenset({"checks", "allowed_labels", "forbidden_terms"})


class FixtureError(ValueError):
    """Raised loudly on any schema violation. A typo'd `expect` key must not
    silently disable a check — a fixture that quietly stops testing anything is
    worse than no fixture, because the scorecard still shows it passing."""


@dataclass(frozen=True)
class Expectation:
    # "all" (every check must pass) or an explicit subset of check names.
    checks: str | list[str]
    allowed_labels: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Fixture:
    id: str
    ticker: str
    close: float
    rsi: float
    macd_hist: float
    gate_reasons: list[str]
    news_text: str
    expect: Expectation
    note: str = ""

    @property
    def is_no_news(self) -> bool:
        return self.news_text.strip().lower() in {"", "none"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def parse_fixture(data: object, source: str = "<string>") -> Fixture:
    _require(isinstance(data, dict), f"{source}: top level must be a mapping")
    assert isinstance(data, dict)  # narrowed by _require, for type checkers

    keys = set(data)
    missing = _REQUIRED_KEYS - keys
    _require(not missing, f"{source}: missing key(s): {', '.join(sorted(missing))}")
    unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
    _require(not unknown, f"{source}: unknown key(s): {', '.join(sorted(unknown))}")

    expect = data["expect"]
    _require(isinstance(expect, dict), f"{source}: `expect` must be a mapping")
    unknown_expect = set(expect) - _EXPECT_KEYS
    _require(
        not unknown_expect,
        f"{source}: unknown `expect` key(s): {', '.join(sorted(unknown_expect))}",
    )
    _require("checks" in expect, f"{source}: `expect.checks` is required")
    checks = expect["checks"]
    _require(
        checks == "all" or (isinstance(checks, list) and all(map(_is_str, checks))),
        f"{source}: `expect.checks` must be 'all' or a list of check names",
    )
    if isinstance(checks, list):
        # A misspelled check name in a subset would silently stop testing what
        # the fixture exists to test, while still reporting a pass.
        unknown_checks = set(checks) - set(CHECK_ORDER)
        _require(
            not unknown_checks,
            f"{source}: unknown check name(s) in `expect.checks`: "
            f"{', '.join(sorted(unknown_checks))} "
            f"(known: {', '.join(CHECK_ORDER)})",
        )

    ticker = data["ticker"]
    _require(
        ticker in SYNTHETIC_TICKERS,
        f"{source}: ticker {ticker!r} is not in SYNTHETIC_TICKERS — fixtures must "
        "never use a real ticker",
    )

    gate_reasons = data["gate_reasons"] or []
    _require(
        isinstance(gate_reasons, list) and all(map(_is_str, gate_reasons)),
        f"{source}: `gate_reasons` must be a list of strings",
    )

    for numeric in ("close", "rsi", "macd_hist"):
        _require(
            isinstance(data[numeric], (int, float))
            and not isinstance(data[numeric], bool),
            f"{source}: `{numeric}` must be a number",
        )
    _require(0.0 <= float(data["rsi"]) <= 100.0, f"{source}: `rsi` must be in 0..100")
    _require(_is_str(data["id"]) and data["id"], f"{source}: `id` must be a string")
    _require(_is_str(data["news_text"]), f"{source}: `news_text` must be a string")

    return Fixture(
        id=data["id"],
        ticker=ticker,
        close=float(data["close"]),
        rsi=float(data["rsi"]),
        macd_hist=float(data["macd_hist"]),
        gate_reasons=list(gate_reasons),
        news_text=data["news_text"],
        note=data.get("note", "") or "",
        expect=Expectation(
            checks=checks,
            allowed_labels=list(expect.get("allowed_labels") or []),
            forbidden_terms=list(expect.get("forbidden_terms") or []),
        ),
    )


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def load_synthetic(directory: Path | None = None) -> list[Fixture]:
    """Every fixture in the directory, sorted by id so a run is ordered the same
    way twice. Raises FixtureError on a duplicate id or any schema violation."""
    directory = directory or SYNTHETIC_DIR
    fixtures = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            fixtures.append(parse_fixture(yaml.safe_load(handle), source=path.name))

    seen: set[str] = set()
    for fixture in fixtures:
        _require(fixture.id not in seen, f"duplicate fixture id: {fixture.id}")
        seen.add(fixture.id)
    return sorted(fixtures, key=lambda f: f.id)
