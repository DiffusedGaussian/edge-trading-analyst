"""Tests over Tier 2: Cohen's kappa and the resumable labelling loop.

The kappa cases are hand-computed. The degenerate one matters most: when a rater
never varies, kappa is undefined, and returning 1.0 there would report a judge
that answers "yes" to everything as being in perfect agreement — the exact
failure this whole tier exists to detect.
"""

from __future__ import annotations

from eval.calibrate import (
    cohens_kappa,
    fetch_human_labels,
    interpret,
    label,
    overlap_pairs,
    render_scores,
    save_human_labels,
    score,
    unlabelled_records,
)

from edge_analyst import store
from edge_analyst.debate import DebateState, DebateTurn, TraderDecision
from edge_analyst.gate import GateResult
from edge_analyst.news_analyst import SentimentSignal

# --- Cohen's kappa ----------------------------------------------------------


def test_kappa_of_perfect_agreement_with_a_varied_rater_is_one():
    pairs = [("yes", "yes"), ("yes", "yes"), ("no", "no"), ("no", "no")]
    result = cohens_kappa(pairs)
    # observed 1.0; chance = 0.5*0.5 + 0.5*0.5 = 0.5; kappa = (1-.5)/(1-.5) = 1.0
    assert result.observed_agreement == 1.0
    assert result.expected_agreement == 0.5
    assert result.kappa == 1.0


def test_kappa_hand_computed_on_a_2x2():
    # human yes/judge yes 5, both-no 3, human-yes/judge-no 1, human-no/judge-yes 1
    pairs = (
        [("yes", "yes")] * 5 + [("no", "no")] * 3 + [("yes", "no")] + [("no", "yes")]
    )
    result = cohens_kappa(pairs)
    # observed = 8/10 = 0.8
    # human_yes = 6/10, judge_yes = 6/10
    # chance = .6*.6 + .4*.4 = 0.52 ; kappa = (0.8-0.52)/(1-0.52) = 0.5833...
    assert result.observed_agreement == 0.8
    assert abs(result.expected_agreement - 0.52) < 1e-12
    assert abs(result.kappa - 0.5833333333333334) < 1e-9
    assert (result.both_yes, result.both_no) == (5, 3)
    assert (result.human_yes_judge_no, result.human_no_judge_yes) == (1, 1)


def test_kappa_is_undefined_when_everyone_always_agrees_on_one_answer():
    """The degenerate all-agree case: chance agreement is 1.0, so the denominator
    is zero. Undefined — not 1.0, which would flatter a judge that never varied."""
    result = cohens_kappa([("yes", "yes")] * 10)
    assert result.kappa is None
    assert result.observed_agreement == 1.0
    assert "undefined" in result.render()


def test_kappa_of_a_yes_to_everything_judge_is_near_zero():
    """The failure Tier 2 exists to catch: high raw agreement, no information.
    Raw agreement is 70%, which looks fine until kappa reports ~0."""
    pairs = [("yes", "yes")] * 7 + [("no", "yes")] * 3
    result = cohens_kappa(pairs)
    assert result.observed_agreement == 0.7
    assert result.kappa == 0.0
    assert "BROKEN RUBRIC" in interpret(result)


def test_kappa_can_be_negative_on_systematic_disagreement():
    pairs = [("yes", "no")] * 5 + [("no", "yes")] * 5
    assert cohens_kappa(pairs).kappa < 0


def test_kappa_of_nothing_is_unknown():
    result = cohens_kappa([])
    assert result.kappa is None
    assert result.n == 0
    assert "n/a" in result.render()
    assert "cannot be interpreted" in interpret(result)


def test_interpret_bands():
    def kappa_of(value):
        # A pair set is not needed to exercise the bands.
        from eval.calibrate import Kappa

        return Kappa(value, 0.9, 0.5, 10, 5, 4, 1, 0)

    assert "BROKEN RUBRIC" in interpret(kappa_of(0.2))
    assert "weak" in interpret(kappa_of(0.5))
    assert "good" in interpret(kappa_of(0.7))
    assert "strong" in interpret(kappa_of(0.9))


# --- overlap ----------------------------------------------------------------


def _human(criterion="news_fidelity", verdict="yes", as_of="t1"):
    return {
        "ticker": "AAPL",
        "as_of": as_of,
        "model": "m",
        "criterion": criterion,
        "verdict": verdict,
        "labelled_at": "now",
    }


def _judge(criterion="news_fidelity", verdict="yes", as_of="t1"):
    return {
        "ticker": "AAPL",
        "as_of": as_of,
        "model": "m",
        "judge_model": "j",
        "criterion": criterion,
        "verdict": verdict,
        "reason": "r",
        "judged_at": "now",
    }


def test_overlap_pairs_groups_by_criterion():
    pairs = overlap_pairs(
        [_human("news_fidelity"), _human("trader_consistent", "no")],
        [_judge("news_fidelity"), _judge("trader_consistent", "no")],
    )
    assert pairs == {
        "news_fidelity": [("yes", "yes")],
        "trader_consistent": [("no", "no")],
    }


def test_overlap_pairs_excludes_unparsed_judge_verdicts():
    """A judge that failed to answer is a parse problem, measured by
    judge_parse_failure_rate — counting it as a disagreement would blame the
    rubric for a formatting failure."""
    assert overlap_pairs([_human()], [_judge(verdict=None)]) == {}


def test_overlap_pairs_ignores_records_only_one_side_covered():
    assert overlap_pairs([_human(as_of="t1")], [_judge(as_of="t2")]) == {}


# --- persistence + labelling loop -------------------------------------------


def _save_decision(conn, as_of, action="buy"):
    store.save_decision(
        conn,
        "AAPL",
        as_of,
        150.0,
        55.0,
        0.5,
        GateResult(material=True, reasons=["price_move"]),
        "some news",
        SentimentSignal("bullish", 7.0, "high", "a fact"),
        TraderDecision(action, "strong case", 150.0, 140.0, 5.0),
        "gemma-3-1b-it",
    )
    store.save_debate_turns(
        conn,
        "AAPL",
        as_of,
        [
            DebateState(
                round=1,
                bull=DebateTurn("buy", "a", "high"),
                bear=DebateTurn("sell", "b", "high"),
            )
        ],
        "gemma-3-1b-it",
    )


def test_human_labels_round_trip(tmp_path):
    conn = store.get_connection(tmp_path / "c.db")
    save_human_labels(conn, [_human()])
    assert fetch_human_labels(conn) == [_human()]


def test_unlabelled_records_skips_already_labelled_criteria(tmp_path):
    conn = store.get_connection(tmp_path / "c.db")
    _save_decision(conn, "t1")
    save_human_labels(conn, [{**_human("news_fidelity"), "as_of": "t1"}])

    pending = unlabelled_records(conn, limit=10)

    assert len(pending) == 1
    _, criteria = pending[0]
    assert "news_fidelity" not in criteria
    assert "trader_consistent" in criteria


def test_label_loop_is_resumable(tmp_path):
    """Never re-present a labelled (ticker, as_of, criterion): a tool that
    re-asks guarantees the calibration set never gets finished."""
    db = tmp_path / "c.db"
    conn = store.get_connection(db)
    _save_decision(conn, "t1")
    conn.close()

    answers = iter(["y", "n", "y", "n"])
    written = label(
        str(db), n=10, input_fn=lambda _: next(answers), print_fn=lambda *_: None
    )
    assert written == 4

    # Second pass has nothing left to ask — the iterator is never touched.
    def explode(_):
        raise AssertionError("re-presented an already-labelled record")

    assert label(str(db), n=10, input_fn=explode, print_fn=lambda *_: None) == 0


def test_label_loop_skips_on_s_without_writing(tmp_path):
    db = tmp_path / "c.db"
    conn = store.get_connection(db)
    _save_decision(conn, "t1")
    conn.close()

    written = label(str(db), n=10, input_fn=lambda _: "s", print_fn=lambda *_: None)
    assert written == 0
    conn = store.get_connection(db)
    assert fetch_human_labels(conn) == []


def test_label_loop_reprompts_on_invalid_input(tmp_path):
    db = tmp_path / "c.db"
    conn = store.get_connection(db)
    _save_decision(conn, "t1")
    conn.close()

    answers = iter(["maybe", "why", "y", "s", "s", "s"])
    written = label(
        str(db), n=10, input_fn=lambda _: next(answers), print_fn=lambda *_: None
    )
    assert written == 1


def test_label_shows_the_same_view_the_judge_sees(tmp_path):
    """Labelling a different rendering would measure disagreement about the
    inputs rather than about the criteria."""
    db = tmp_path / "c.db"
    conn = store.get_connection(db)
    _save_decision(conn, "t1")
    conn.close()

    printed: list[str] = []
    label(
        str(db),
        n=10,
        input_fn=lambda _: "s",
        print_fn=lambda *args: printed.append(" ".join(str(a) for a in args)),
    )
    text = "\n".join(printed)
    assert "RSI: 55.0 (neutral)" in text
    assert "Trader decision: buy" in text
    assert "Round 1 bull" in text


# --- scoring end to end -----------------------------------------------------


def test_score_and_render_end_to_end(tmp_path):
    db = tmp_path / "c.db"
    conn = store.get_connection(db)
    save_human_labels(
        conn,
        [
            _human("news_fidelity", "yes", "t1"),
            _human("news_fidelity", "no", "t2"),
        ],
    )
    store.save_criterion_verdicts(
        conn,
        [
            _judge("news_fidelity", "yes", "t1"),
            _judge("news_fidelity", "no", "t2"),
        ],
    )
    conn.close()

    scores = score(str(db))
    assert scores["news_fidelity"].kappa == 1.0
    text = render_scores(scores)
    assert "news_fidelity" in text
    assert "confusion" in text
    assert "reading" in text


def test_render_scores_with_no_overlap_explains_what_to_do(tmp_path):
    assert "no overlap" in render_scores({})
