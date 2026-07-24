"""LLM-as-judge rubric: prompt builder + forgiving parser for scoring a
persisted decision's reasoning quality (not trading quality — whether the
call would have been profitable is a separate concern, belonging to Phase 5's
paper-trade P&L loop, not this judge).

Reuses the project's existing sentinel key-value + forgiving-parser pattern
(edge_analyst.llm_parsing.extract_field) rather than JSON, for consistency
with analyst.py/debate.py, and edge_analyst.indicators.format_market_context
so the judge sees indicator labels built the exact same way the cascade's own
prompts do — no separate labeling logic to drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

from edge_analyst.indicators import format_market_context
from edge_analyst.llm_parsing import extract_field

_SYSTEM_PROMPT = """You are auditing an automated trading analyst's reasoning, \
not its profitability. You are given the real technical indicators, news, a \
bull/bear debate, and the final trader decision for one cycle. Judge only \
whether the reasoning is internally consistent and well-grounded. Respond in \
EXACTLY this format, nothing else before or after it:

BULL_BEAR_DISTINCT: <yes or no — did bull and bear argue genuinely different \
positions, or did one just echo the other's point?>
INDICATOR_CONSISTENT: <yes or no — does the trader's reasoning match the \
given RSI/MACD labels, with no contradictions?>
NEWS_FIDELITY: <yes or no — is the news represented accurately, with nothing \
fabricated?>
TRADER_CONSISTENT: <yes or no — does the final action logically follow from \
the debate's bull/bear positions?>
OVERALL_SCORE: <0-10>
NOTES: <1-2 sentences; if any field above is "no", name which one and why>"""


def _format_debate_turns(turns: list[dict]) -> str:
    if not turns:
        return "No debate rounds recorded."
    lines = []
    for turn in turns:
        lines.append(
            f"Round {turn['round']} {turn['side']}: {turn['stance']} "
            f"({turn['confidence']}) — {turn['key_point']}"
        )
    return "\n".join(lines)


def build_judge_prompt(record: dict) -> list[dict]:
    gate_reasons = [r for r in (record.get("gate_reasons") or "").split(",") if r]
    context = format_market_context(
        record["ticker"],
        record["close"],
        record["rsi"],
        record["macd_hist"],
        gate_reasons,
    )
    user_prompt = (
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
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class Judgment:
    bull_bear_distinct: str
    indicator_consistent: str
    news_fidelity: str
    trader_consistent: str
    overall_score: float
    notes: str


_YES_NO = {"yes", "no"}


def _parse_yes_no(text: str, field: str) -> str:
    value = (extract_field(text, field) or "").lower()
    return value if value in _YES_NO else "unknown"


def parse_judgment(text: str) -> Judgment:
    """Same forgiving line-scan pattern as the rest of the cascade's parsers
    (news_analyst.py, debate.py) — hard defaults on any miss, never raises,
    since a malformed judge response shouldn't crash a batch judging run."""
    score_raw = extract_field(text, "OVERALL_SCORE")
    try:
        score = max(0.0, min(10.0, float(score_raw)))
    except (TypeError, ValueError):
        score = 5.0

    return Judgment(
        bull_bear_distinct=_parse_yes_no(text, "BULL_BEAR_DISTINCT"),
        indicator_consistent=_parse_yes_no(text, "INDICATOR_CONSISTENT"),
        news_fidelity=_parse_yes_no(text, "NEWS_FIDELITY"),
        trader_consistent=_parse_yes_no(text, "TRADER_CONSISTENT"),
        overall_score=score,
        notes=extract_field(text, "NOTES") or "no notes parsed",
    )
