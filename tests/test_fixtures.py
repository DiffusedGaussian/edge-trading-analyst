"""Tests over the synthetic fixture set and its loader.

The loader is strict on purpose: a fixture that silently stops testing what it
was written to test is worse than no fixture, because the scorecard still
reports it green. These tests pin that strictness.
"""

from __future__ import annotations

import pytest
import yaml
from eval.checks import CHECK_ORDER
from eval.fixtures import (
    SYNTHETIC_TICKERS,
    FixtureError,
    load_synthetic,
    parse_fixture,
)

# Each of the 16 encodes one specific failure mode; the list is the deliverable,
# so it is pinned here rather than merely counted.
_EXPECTED_IDS = {
    "rsi_neutral_bland_news",
    "rsi_overbought_good_news",
    "rsi_oversold_bad_news",
    "macd_bullish_bearish_news",
    "macd_flat",
    "no_news_at_all",
    "irrelevant_news",
    "specific_number_in_news",
    "two_contradictory_headlines",
    "very_long_news",
    "prompt_injection_sentinel",
    "prompt_injection_instruction",
    "empty_gate_reasons",
    "extreme_rsi_100",
    "extreme_rsi_0",
    "non_ascii_news",
}


def test_all_sixteen_fixtures_load():
    fixtures = load_synthetic()
    assert len(fixtures) == 16
    assert {f.id for f in fixtures} == _EXPECTED_IDS


def test_ids_are_unique_and_match_their_filename():
    fixtures = load_synthetic()
    assert len({f.id for f in fixtures}) == len(fixtures)


def test_load_is_ordered_deterministically():
    assert [f.id for f in load_synthetic()] == sorted(f.id for f in load_synthetic())


def test_every_ticker_is_synthetic():
    """A real ticker here would let synthetic output be mistaken for a replayed
    real day — the one boundary the two input sets must never cross."""
    for fixture in load_synthetic():
        assert fixture.ticker in SYNTHETIC_TICKERS, fixture.id


def test_every_fixture_has_a_note_explaining_what_it_probes():
    for fixture in load_synthetic():
        assert fixture.note.strip(), fixture.id


def test_declared_check_subsets_name_real_checks():
    for fixture in load_synthetic():
        if isinstance(fixture.expect.checks, list):
            assert set(fixture.expect.checks) <= set(CHECK_ORDER), fixture.id


def test_specific_fixtures_encode_their_probe():
    by_id = {f.id: f for f in load_synthetic()}

    # The observed 2026-07-23 regression: neutral RSI against the 30/70 bands.
    regression = by_id["rsi_neutral_bland_news"]
    assert regression.rsi == 61.7
    assert "overbought" in regression.expect.forbidden_terms

    assert by_id["macd_flat"].macd_hist == 0.0
    assert by_id["no_news_at_all"].is_no_news
    assert by_id["empty_gate_reasons"].gate_reasons == []
    assert by_id["extreme_rsi_100"].rsi == 100.0
    assert by_id["extreme_rsi_0"].rsi == 0.0
    assert len(by_id["very_long_news"].news_text) > 2500
    assert "LABEL: bullish" in by_id["prompt_injection_sentinel"].news_text
    assert "Ignore previous instructions" in (
        by_id["prompt_injection_instruction"].news_text
    )
    assert "Zürich" in by_id["non_ascii_news"].news_text


_VALID = """
id: probe
ticker: TESTA
close: 100.0
rsi: 50.0
macd_hist: 0.0
gate_reasons: []
news_text: "something"
expect:
  checks: all
"""


def _parse(text: str):
    return parse_fixture(yaml.safe_load(text))


def test_a_valid_minimal_fixture_parses():
    fixture = _parse(_VALID)
    assert fixture.id == "probe"
    assert fixture.expect.allowed_labels == []


def test_unknown_top_level_key_raises():
    with pytest.raises(FixtureError, match="unknown key"):
        _parse(_VALID + "\nunexpected: 1\n")


def test_typoed_expect_key_raises():
    """The failure this strictness exists for: `forbidden_term` (singular) would
    otherwise parse fine and quietly forbid nothing."""
    with pytest.raises(FixtureError, match="unknown `expect` key"):
        _parse(_VALID.replace("  checks: all", "  checks: all\n  forbidden_term: [x]"))


def test_unknown_check_name_raises():
    with pytest.raises(FixtureError, match="unknown check name"):
        _parse(_VALID.replace("  checks: all", "  checks: [feilds_parsed]"))


def test_missing_required_key_raises():
    with pytest.raises(FixtureError, match="missing key"):
        _parse(_VALID.replace("rsi: 50.0\n", ""))


def test_real_ticker_raises():
    with pytest.raises(FixtureError, match="SYNTHETIC_TICKERS"):
        _parse(_VALID.replace("TESTA", "AAPL"))


def test_out_of_range_rsi_raises():
    with pytest.raises(FixtureError, match="0..100"):
        _parse(_VALID.replace("rsi: 50.0", "rsi: 140.0"))


def test_non_numeric_close_raises():
    with pytest.raises(FixtureError, match="must be a number"):
        _parse(_VALID.replace("close: 100.0", 'close: "cheap"'))


def test_missing_expect_checks_raises():
    with pytest.raises(FixtureError, match="expect.checks"):
        _parse(_VALID.replace("  checks: all", "  allowed_labels: [neutral]"))


def test_non_mapping_top_level_raises():
    with pytest.raises(FixtureError, match="must be a mapping"):
        parse_fixture(["not", "a", "mapping"])


def test_duplicate_ids_raise(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(_VALID, encoding="utf-8")
    with pytest.raises(FixtureError, match="duplicate fixture id"):
        load_synthetic(tmp_path)
