"""Tier 2: a human-labelled calibration set, and Cohen's kappa between those
labels and the judge's.

This is last in the harness and the thing that makes the rest of it trustworthy.
Without it every Tier 1 number is unfalsifiable: a judge that answers "yes" to
everything and a cascade that is genuinely good produce the same scorecard, and
nothing in Tiers 0-1 can tell them apart.

    python -m eval.calibrate label --n 40     # label by hand, resumable
    python -m eval.calibrate score            # kappa vs the judge, per criterion

Kappa rather than raw agreement, because raw agreement is inflated by whatever the
base rate happens to be: on a criterion where 90% of records are genuinely "yes",
a judge that always says yes scores 90% agreement and has learned nothing. Kappa
measures agreement *above chance*, so it collapses to ~0 for exactly that judge.

Interpretation is printed with the numbers, not left in the docs:

    kappa < 0.4  -> the rubric wording is broken, NOT the cascade. Fix
                    CRITERIA[criterion] in eval/rubric.py and re-judge before
                    drawing any conclusion about a model.

~15 lines of stdlib. Deliberately no scipy/sklearn: the whole point of the
dependency discipline here is that eval adds no runtime weight, and a 2x2
agreement statistic does not justify a 60MB wheel.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass

from edge_analyst import store
from eval.rubric import CRITERION_NAMES, format_record

_VALID_INPUT = {"y": "yes", "n": "no", "s": None}


# --- Cohen's kappa -----------------------------------------------------------


@dataclass(frozen=True)
class Kappa:
    kappa: float | None
    observed_agreement: float | None
    expected_agreement: float | None
    n: int
    # Confusion counts, so a bad kappa can be diagnosed rather than just noticed.
    both_yes: int
    both_no: int
    human_yes_judge_no: int
    human_no_judge_yes: int

    def render(self) -> str:
        if self.kappa is None:
            if self.n == 0:
                return "n/a (no overlapping labels)"
            # Every label on one side is identical, so chance agreement is 1.0
            # and kappa's denominator is zero. Undefined, not perfect.
            return (
                f"undefined ({self.n} pairs, but one rater never varied — "
                f"agreement {self.observed_agreement:.0%})"
            )
        return (
            f"{self.kappa:+.2f}  (n={self.n}, observed {self.observed_agreement:.0%}, "
            f"chance {self.expected_agreement:.0%})"
        )


def cohens_kappa(pairs: list[tuple[str, str]]) -> Kappa:
    """Cohen's kappa over (human, judge) yes/no pairs.

    Returns kappa=None when it is genuinely undefined — no pairs at all, or one
    rater giving a single constant answer, which makes expected agreement 1.0 and
    the denominator zero. Returning 1.0 there (the tempting shortcut) would report
    a rater who never varied as being in perfect agreement.
    """
    n = len(pairs)
    if n == 0:
        return Kappa(None, None, None, 0, 0, 0, 0, 0)

    both_yes = sum(1 for h, j in pairs if h == "yes" and j == "yes")
    both_no = sum(1 for h, j in pairs if h == "no" and j == "no")
    human_yes_judge_no = sum(1 for h, j in pairs if h == "yes" and j == "no")
    human_no_judge_yes = sum(1 for h, j in pairs if h == "no" and j == "yes")

    observed = (both_yes + both_no) / n
    human_yes = (both_yes + human_yes_judge_no) / n
    judge_yes = (both_yes + human_no_judge_yes) / n
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)

    kappa = None if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Kappa(
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        n=n,
        both_yes=both_yes,
        both_no=both_no,
        human_yes_judge_no=human_yes_judge_no,
        human_no_judge_yes=human_no_judge_yes,
    )


def interpret(kappa: Kappa) -> str:
    """The reading goes next to the number, so nobody has to remember the bands."""
    if kappa.kappa is None:
        return "cannot be interpreted — label more, or more varied, records"
    if kappa.kappa < 0.4:
        return (
            "BROKEN RUBRIC: the judge and you are barely agreeing above chance. "
            "Fix this criterion's wording in eval/rubric.py CRITERIA and re-judge "
            "before concluding anything about a model."
        )
    if kappa.kappa < 0.6:
        return "weak: usable for coarse ranking, not for small differences"
    if kappa.kappa < 0.8:
        return "good: the judge is a reasonable proxy for your reading"
    return "strong agreement"


# --- labelling ---------------------------------------------------------------


def unlabelled_records(
    conn, limit: int, model: str | None = None
) -> list[tuple[dict, list[str]]]:
    """(record, criteria still needing a label) for records with any gap.

    Resumability is a hard requirement rather than a nicety: hand-labelling 40
    records is one or two sittings, and a tool that re-presents finished work
    guarantees the set never gets finished.
    """
    labelled = {
        (row["ticker"], row["as_of"], row["criterion"])
        for row in fetch_human_labels(conn)
    }
    pending = []
    for record in store.fetch_decisions_for_judging(conn, limit=limit, model=model):
        missing = [
            criterion
            for criterion in CRITERION_NAMES
            if (record["ticker"], record["as_of"], criterion) not in labelled
        ]
        if missing:
            pending.append((record, missing))
    return pending


def label(
    db_path: str,
    n: int = 40,
    model: str | None = None,
    input_fn=input,
    print_fn=print,
) -> int:
    """Walk unlabelled decisions one at a time. Returns the number of labels
    written. `input_fn`/`print_fn` are injected so the loop is testable."""
    conn = store.get_connection(db_path)
    pending = unlabelled_records(conn, n, model)
    if not pending:
        print_fn("nothing left to label")
        conn.close()
        return 0

    written = 0
    for index, (record, criteria) in enumerate(pending, start=1):
        print_fn(f"\n{'=' * 70}")
        print_fn(f"[{index}/{len(pending)}] {record['ticker']} {record['as_of']}")
        print_fn(f"model: {record.get('model') or 'unattributed'}")
        # Exactly what the judge is shown — labelling a different view would
        # measure a disagreement about the inputs, not about the criteria.
        print_fn(format_record(record))
        print_fn("")

        rows = []
        for criterion in criteria:
            answer = None
            while answer is None:
                raw = input_fn(f"  {criterion}? [y/n/s(kip)] ").strip().lower()
                if raw in _VALID_INPUT:
                    answer = raw
                    break
                print_fn("    answer y, n, or s")
            if _VALID_INPUT[answer] is None:
                continue
            rows.append(
                {
                    "ticker": record["ticker"],
                    "as_of": record["as_of"],
                    "model": record.get("model"),
                    "criterion": criterion,
                    "verdict": _VALID_INPUT[answer],
                    "labelled_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
        if rows:
            save_human_labels(conn, rows)
            written += len(rows)

    conn.close()
    print_fn(f"\nwrote {written} label(s)")
    return written


# --- persistence -------------------------------------------------------------

_HUMAN_LABEL_COLUMNS = [
    "ticker",
    "as_of",
    "model",
    "criterion",
    "verdict",
    "labelled_at",
]

_HUMAN_LABELS_SCHEMA = """
CREATE TABLE IF NOT EXISTS human_labels (
    ticker TEXT, as_of TEXT, model TEXT,
    criterion TEXT, verdict TEXT, labelled_at TEXT,
    PRIMARY KEY (ticker, as_of, criterion)
);
"""


def _ensure_table(conn) -> None:
    """Created here rather than in store.SCHEMA: human_labels is eval-only and
    hand-written, and the runtime store has no business knowing about it."""
    conn.executescript(_HUMAN_LABELS_SCHEMA)
    conn.commit()


def save_human_labels(conn, rows: list[dict]) -> None:
    _ensure_table(conn)
    conn.executemany(
        f"""INSERT OR REPLACE INTO human_labels
            ({", ".join(_HUMAN_LABEL_COLUMNS)})
            VALUES ({", ".join("?" * len(_HUMAN_LABEL_COLUMNS))})""",
        [tuple(row.get(column) for column in _HUMAN_LABEL_COLUMNS) for row in rows],
    )
    conn.commit()


def fetch_human_labels(conn) -> list[dict]:
    _ensure_table(conn)
    rows = conn.execute(
        f"""SELECT {", ".join(_HUMAN_LABEL_COLUMNS)} FROM human_labels
            ORDER BY ticker, as_of, criterion"""
    ).fetchall()
    return [dict(zip(_HUMAN_LABEL_COLUMNS, row, strict=True)) for row in rows]


# --- scoring -----------------------------------------------------------------


def overlap_pairs(
    human: list[dict], judge: list[dict]
) -> dict[str, list[tuple[str, str]]]:
    """Per criterion, the (human, judge) pairs where both sides have a real
    yes/no. A judge verdict of None is excluded rather than counted as a
    disagreement — the judge failing to answer is a parse problem, measured
    separately by judge_parse_failure_rate."""
    judge_by_key = {
        (row["ticker"], row["as_of"], row["criterion"]): row["verdict"]
        for row in judge
        if row["verdict"] is not None
    }
    pairs: dict[str, list[tuple[str, str]]] = {}
    for row in human:
        if row["verdict"] is None:
            continue
        key = (row["ticker"], row["as_of"], row["criterion"])
        judge_verdict = judge_by_key.get(key)
        if judge_verdict is not None:
            pairs.setdefault(row["criterion"], []).append(
                (row["verdict"], judge_verdict)
            )
    return pairs


def score(db_path: str, judge_model: str | None = None) -> dict[str, Kappa]:
    conn = store.get_connection(db_path)
    human = fetch_human_labels(conn)
    judge = store.fetch_criterion_verdicts(conn, judge_model=judge_model)
    conn.close()
    return {
        criterion: cohens_kappa(pairs)
        for criterion, pairs in sorted(overlap_pairs(human, judge).items())
    }


def render_scores(scores: dict[str, Kappa]) -> str:
    if not scores:
        return (
            "no overlap between human labels and judge verdicts yet.\n"
            "run `python -m eval.calibrate label` and `make eval-judge` over the "
            "same decisions first."
        )
    lines = ["Cohen's kappa, human vs judge, per criterion:"]
    for criterion, kappa in scores.items():
        lines.append(f"\n  {criterion}")
        lines.append(f"    kappa      {kappa.render()}")
        lines.append(
            f"    confusion  both-yes {kappa.both_yes}, both-no {kappa.both_no}, "
            f"human-yes/judge-no {kappa.human_yes_judge_no}, "
            f"human-no/judge-yes {kappa.human_no_judge_yes}"
        )
        lines.append(f"    reading    {interpret(kappa)}")
    lines.append(
        "\nRe-run this whenever the judge model or any criterion prompt changes — "
        "a rubric edit invalidates the previous kappa."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tier 2 human calibration")
    sub = parser.add_subparsers(dest="command", required=True)

    label_cmd = sub.add_parser("label", help="hand-label decisions (resumable)")
    label_cmd.add_argument("--n", type=int, default=40)
    label_cmd.add_argument("--db", default="data/edge_analyst.db")
    label_cmd.add_argument("--model", default="", help="label one model's decisions")

    score_cmd = sub.add_parser("score", help="Cohen's kappa vs the judge")
    score_cmd.add_argument("--db", default="data/edge_analyst.db")
    score_cmd.add_argument("--judge-model", default="")

    args = parser.parse_args(argv)
    if args.command == "label":
        label(args.db, args.n, args.model or None)
    else:
        print(render_scores(score(args.db, args.judge_model or None)))


if __name__ == "__main__":
    main()
