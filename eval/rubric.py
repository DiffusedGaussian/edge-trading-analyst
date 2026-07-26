"""Tier 1 LLM-as-judge: prompt builders and parsers for scoring a persisted
decision's *reasoning* quality — not its trading quality, which belongs to
Phase 5's paper-trade P&L loop, not this judge.

Three deliberate departures from the first version of this file:

1. **One criterion per call.** A four-question prompt makes the model's answer to
   question 4 depend on what it said about question 1. Separate calls make each
   verdict independent, and on Modal they batch into the same llm.chat(), so the
   wall-clock cost is close to free.
2. **No imputation, ever.** An unparseable judge response yields
   `verdict=None`, not a default. A run with a high parse-failure rate is not a
   low-scoring run — it is an *invalid* run, and averaging a made-up 5.0 into
   real scores destroys exactly that distinction.
3. **No OVERALL_SCORE.** An LLM's absolute 0-10 score is low-resolution and
   clusters at 7-8. eval/report.py derives `yes_count / answered_count` instead,
   and reports the parse-failure rate beside it as a separate number.

Pairwise comparison (build_pairwise_prompt) is the mode a bake-off actually
wants: judges are markedly more reliable at "which of these two is better" than
at "score this out of ten". Every pair must be judged in both orders — the
resulting order_flip_rate is the judge's measured noise floor, and any win-rate
gap smaller than it is not a result.

Reuses the project's sentinel key-value + forgiving-parser pattern
(edge_analyst.llm_parsing.extract_field) and edge_analyst.indicators.
format_market_context, so the judge sees indicator labels built exactly the way
the cascade's own prompts build them — no second labelling path to drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from edge_analyst.indicators import format_market_context
from edge_analyst.llm_parsing import extract_field

_RESPONSE_FORMAT = """Respond in EXACTLY this format, nothing else before or \
after it:

VERDICT: <yes or no>
REASON: <one sentence>"""

_PREAMBLE = """You are auditing an automated trading analyst's reasoning, not \
its profitability. You are given the real technical indicators, the news, a \
bull/bear debate, and the final trader decision for one cycle. Judge ONLY the \
single question below — ignore every other aspect of the record, however \
wrong it looks."""

# One focused prompt per criterion, each with a worked yes and a worked no. The
# examples are what stop a judge from answering the question it wishes it had
# been asked; without them every criterion drifts toward "is this good analysis
# overall", and four criteria collapse into one.
CRITERIA: dict[str, str] = {
    "bull_bear_distinct": f"""{_PREAMBLE}

QUESTION: Did the bull and the bear argue genuinely different positions, or did \
one simply restate the other's point in different words?

Answer "yes" if their key points rest on different evidence or reach different \
conclusions. Answer "no" if one side's key point is a paraphrase of the other's, \
or if both cite the same single fact for the same conclusion.

Example (yes):
Bull cites a 12% revenue rise; bear cites the overbought RSI. Different \
evidence, opposing conclusions.
VERDICT: yes
REASON: The two sides rest on different evidence and reach opposing conclusions.

Example (no):
Bull says "momentum is strong"; bear says "momentum looks strong but that is \
risky". Same fact, same reading, one hedged.
VERDICT: no
REASON: The bear restates the bull's momentum claim rather than opposing it.

{_RESPONSE_FORMAT}""",
    "indicator_consistent": f"""{_PREAMBLE}

QUESTION: Does the trader's reasoning agree with the RSI and MACD labels given \
in the indicator block, with no contradiction?

The labels in the indicator block are computed deterministically and are ground \
truth. Answer "no" if the reasoning describes the indicators in a way that \
contradicts those labels, even where its overall conclusion is defensible. \
Disagreeing with the *weight* of an indicator is fine; misdescribing its \
*value* is not.

Example (yes):
RSI 61.7 (neutral). Reasoning: "RSI is mid-range, so the news carries the call."
VERDICT: yes
REASON: The reasoning describes RSI as mid-range, matching the neutral label.

Example (no):
RSI 61.7 (neutral). Reasoning: "The stock is overbought, so we should trim."
VERDICT: no
REASON: It calls a neutral RSI overbought, contradicting the given label.

{_RESPONSE_FORMAT}""",
    "news_fidelity": f"""{_PREAMBLE}

QUESTION: Is the news represented accurately, with nothing invented?

Answer "no" if any claim about the news is absent from the news text, if a \
figure is altered, or if an event is asserted where the news text is empty or \
says "none". Omitting part of the news is acceptable; adding to it is not.

Example (yes):
News: "revenue rose 12%". Rationale: "the 12% revenue rise supports the case."
VERDICT: yes
REASON: The only news claim matches the news text exactly.

Example (no):
News: "none". Rationale: "strong earnings this morning support the case."
VERDICT: no
REASON: It asserts an earnings event that does not appear in the input at all.

{_RESPONSE_FORMAT}""",
    "trader_consistent": f"""{_PREAMBLE}

QUESTION: Does the final trader action follow logically from the bull and bear \
positions as recorded?

Answer "yes" if the action is a defensible resolution of the two positions, \
including a Hold on a genuine standoff. Answer "no" if the action contradicts \
the side the reasoning itself endorses, or if it rests on a position neither \
analyst took.

Example (yes):
Bull: buy. Bear: sell. Action: Hold, reasoning cites unresolved disagreement.
VERDICT: yes
REASON: Hold is a defensible resolution of an unresolved buy-versus-sell split.

Example (no):
Bull: buy (high confidence). Bear: hold (low). Action: Sell, reasoning endorses \
the bull case.
VERDICT: no
REASON: It sells while its own reasoning endorses the bull's buy case.

{_RESPONSE_FORMAT}""",
}

CRITERION_NAMES: tuple[str, ...] = tuple(CRITERIA)


def _format_debate_turns(turns: list[dict]) -> str:
    if not turns:
        return "No debate rounds recorded."
    return "\n".join(
        f"Round {turn['round']} {turn['side']}: {turn['stance']} "
        f"({turn['confidence']}) — {turn['key_point']}"
        for turn in turns
    )


def format_record(record: dict) -> str:
    """The record as the judge sees it. One renderer, shared by the single-record
    and pairwise prompts, so the two modes can never show different views of the
    same decision."""
    gate_reasons = [r for r in (record.get("gate_reasons") or "").split(",") if r]
    context = format_market_context(
        record["ticker"],
        record["close"],
        record["rsi"],
        record["macd_hist"],
        gate_reasons,
    )
    return (
        f"{context}\n"
        f"News: {record.get('news_text') or 'none'}\n"
        f"Sentiment: {record.get('sentiment_label')} "
        f"(score={record.get('sentiment_score')}, "
        f"confidence={record.get('sentiment_confidence')}) — "
        f"{record.get('sentiment_rationale')}\n"
        f"Debate:\n{_format_debate_turns(record.get('debate_turns', []))}\n"
        f"Trader decision: {record.get('trader_action')} — "
        f"{record.get('trader_reasoning')} "
        f"(entry={record.get('trader_entry_price')}, "
        f"stop={record.get('trader_stop_loss')}, "
        f"sizing={record.get('trader_position_sizing')})"
    )


def build_judge_prompt(record: dict, criterion: str) -> list[dict]:
    if criterion not in CRITERIA:
        raise KeyError(f"unknown criterion: {criterion}")
    return [
        {"role": "system", "content": CRITERIA[criterion]},
        {"role": "user", "content": format_record(record)},
    ]


@dataclass(frozen=True)
class CriterionVerdict:
    criterion: str
    # "yes" | "no" | None. None means the judge's response could not be parsed —
    # never a default. Imputing a value here is what left the previous version
    # unable to tell a bad run from an invalid one.
    verdict: str | None
    reason: str | None


_YES_NO = ("yes", "no")


def parse_verdict(text: str, criterion: str) -> CriterionVerdict:
    raw = (extract_field(text, "VERDICT") or "").strip().lower()
    # Tolerate "yes." / "yes, because ...": a judge answering the right question
    # in a slightly wrong shape is not a parse failure. Anything else is.
    verdict = next((value for value in _YES_NO if raw.startswith(value)), None)
    return CriterionVerdict(
        criterion=criterion,
        verdict=verdict,
        reason=extract_field(text, "REASON"),
    )


# --- pairwise ----------------------------------------------------------------

_PAIRWISE_FORMAT = """Respond in EXACTLY this format, nothing else before or \
after it:

WINNER: <A or B or tie>
REASON: <one sentence>"""

_PAIRWISE_PREAMBLE = """You are comparing two automated trading analysts' \
reasoning about the same stock on the same day. They saw identical inputs. Judge \
ONLY the single question below, and judge the reasoning, not the profitability. \
If they are genuinely equivalent on this question, answer tie — do not \
manufacture a difference."""

_PAIRWISE_QUESTIONS = {
    "bull_bear_distinct": "Which record's bull and bear argued more genuinely "
    "different positions, rather than one restating the other?",
    "indicator_consistent": "Which record's reasoning agrees better with the "
    "given RSI and MACD labels, with fewer contradictions?",
    "news_fidelity": "Which record represents the news more accurately, with "
    "less invented or altered detail?",
    "trader_consistent": "Which record's final action follows more logically "
    "from its own bull and bear positions?",
}

ORDERS: tuple[str, ...] = ("ab", "ba")


def build_pairwise_prompt(
    record_a: dict, record_b: dict, criterion: str, order: str
) -> list[dict]:
    """`order` is "ab" or "ba". The same pair must be judged both ways: judges
    carry a position bias, and the fraction of pairs whose winner changes when
    the order is swapped (order_flip_rate) is the noise floor below which no
    win-rate difference means anything."""
    if criterion not in _PAIRWISE_QUESTIONS:
        raise KeyError(f"unknown criterion: {criterion}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")

    # "A" and "B" label *positions*, not records: in "ba" order record_b is shown
    # first and is therefore A. resolve_pairwise_winner maps back.
    first, second = (record_a, record_b) if order == "ab" else (record_b, record_a)
    system = (
        f"{_PAIRWISE_PREAMBLE}\n\nQUESTION: "
        f"{_PAIRWISE_QUESTIONS[criterion]}\n\n{_PAIRWISE_FORMAT}"
    )
    user = (
        f"=== RECORD A ===\n{format_record(first)}\n\n"
        f"=== RECORD B ===\n{format_record(second)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True)
class PairwiseVerdict:
    criterion: str
    order: str
    # "A" | "B" | "tie" | None — the position shown, not the record. None on an
    # unparseable response, never a default.
    winner: str | None
    reason: str | None


_WINNERS = {"a": "A", "b": "B", "tie": "tie"}


def parse_pairwise(text: str, criterion: str, order: str) -> PairwiseVerdict:
    raw = (extract_field(text, "WINNER") or "").strip().lower()
    winner = _WINNERS.get(raw)
    if winner is None:
        # Tolerate "Record A" / "Record B" without accepting any prose that
        # merely happens to contain an "a".
        for token, value in (("record a", "A"), ("record b", "B")):
            if raw.startswith(token):
                winner = value
                break
    return PairwiseVerdict(
        criterion=criterion,
        order=order,
        winner=winner,
        reason=extract_field(text, "REASON"),
    )


def resolve_pairwise_winner(verdict: PairwiseVerdict) -> str | None:
    """Maps a position winner ("A"/"B") onto the record it refers to
    ("model_a"/"model_b"), undoing the display order."""
    if verdict.winner is None or verdict.winner == "tie":
        return verdict.winner
    if verdict.order == "ab":
        return "model_a" if verdict.winner == "A" else "model_b"
    return "model_b" if verdict.winner == "A" else "model_a"


def order_flip(verdict_ab: PairwiseVerdict, verdict_ba: PairwiseVerdict) -> bool | None:
    """True when swapping the display order changed which *record* won. None when
    either direction was unparseable — unknown, not "no flip"."""
    first = resolve_pairwise_winner(verdict_ab)
    second = resolve_pairwise_winner(verdict_ba)
    if first is None or second is None:
        return None
    return first != second


# --- judge-family bias -------------------------------------------------------

# A judge scoring output from its own family self-prefers. Keeping the mapping
# explicit lets report.py warn, rather than relying on the reader to notice that
# Qwen2.5-32B is judging Qwen3-30B-A3B.
_FAMILY_MARKERS = ("qwen", "llama", "mistral", "gemma", "phi", "olmo", "granite")


def model_family(model_name: str) -> str | None:
    lowered = (model_name or "").lower()
    return next((marker for marker in _FAMILY_MARKERS if marker in lowered), None)


def same_family(model_name: str, judge_model: str) -> bool:
    """Whether a judgment is at risk of same-family self-preference."""
    family = model_family(model_name)
    return family is not None and family == model_family(judge_model)
