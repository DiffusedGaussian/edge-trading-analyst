"""Phase 4 bull/bear debate loop. Carries a fixed-size DebateState between
rounds (overwritten each round, not appended to) rather than a growing
transcript — avoids TradingAgents' quadratic context-resend pattern while
keeping the round-to-round rebuttal fully intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .indicators import format_market_context
from .llm_parsing import extract_field

if TYPE_CHECKING:
    # Type-only: llm_client stays a per-call local import in this module (the
    # functions below), which is what lets tests patch the HTTP boundary.
    from .llm_client import GenSettings

_STANCES = {"buy", "hold", "sell"}
_CONFIDENCES = {"low", "medium", "high"}

_DEBATE_FORMAT = """Respond in EXACTLY this format, nothing else before or after it:

STANCE: <Buy, Hold, or Sell>
KEY_POINT: <one sentence, the single strongest fact supporting your stance>
CONFIDENCE: <low, medium, or high>"""

_BULL_SYSTEM_PROMPT = f"""You are a Bull analyst. Your assigned position is to \
argue for buying or holding this stock, but you must be honest — if the \
evidence or the rebuttal below genuinely undermines your case, soften your \
stance toward Hold rather than defend a losing argument. You are given real, \
already-computed technical indicators and a news item — treat them as ground \
truth, do not recompute or second-guess them.

{_DEBATE_FORMAT}"""

_BEAR_SYSTEM_PROMPT = f"""You are a Bear analyst. Your assigned position is to \
argue for selling or avoiding this stock, but you must be honest — if the \
evidence or the rebuttal below genuinely undermines your case, soften your \
stance toward Hold rather than defend a losing argument. You are given real, \
already-computed technical indicators and a news item — treat them as ground \
truth, do not recompute or second-guess them.

{_DEBATE_FORMAT}"""


def build_debate_prompt(
    persona: str,
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
    news_text: str,
    opposing_key_point: str | None,
) -> list[dict]:
    system_prompt = _BULL_SYSTEM_PROMPT if persona == "bull" else _BEAR_SYSTEM_PROMPT
    context = format_market_context(ticker, close, rsi_value, macd_hist, fired_reasons)
    user_prompt = f"{context}\nNews: {news_text}"
    if opposing_key_point:
        opponent = "Bear" if persona == "bull" else "Bull"
        user_prompt += f"\n{opponent}'s current strongest point: {opposing_key_point}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class DebateTurn:
    stance: str
    key_point: str
    confidence: str
    # Which sentinels fell back to a default — see SentimentSignal.fallbacks.
    fallbacks: frozenset[str] = frozenset()


def parse_debate_response(text: str) -> DebateTurn:
    """Same forgiving line-scan pattern as news_analyst.py — hard defaults on any
    parse miss (Hold/low/placeholder), never raises."""
    fallbacks: set[str] = set()

    stance = (extract_field(text, "STANCE") or "").lower()
    if stance not in _STANCES:
        stance = "hold"
        fallbacks.add("STANCE")

    confidence = (extract_field(text, "CONFIDENCE") or "").lower()
    if confidence not in _CONFIDENCES:
        confidence = "low"
        fallbacks.add("CONFIDENCE")

    key_point = extract_field(text, "KEY_POINT")
    if not key_point:
        key_point = "no key point parsed"
        fallbacks.add("KEY_POINT")

    return DebateTurn(
        stance=stance,
        key_point=key_point,
        confidence=confidence,
        fallbacks=frozenset(fallbacks),
    )


@dataclass
class DebateState:
    round: int
    bull: DebateTurn
    bear: DebateTurn


def should_continue_debate(state: DebateState) -> bool:
    """Only escalate to another round on a genuine Buy-vs-Sell standoff. If
    either side has already moved to Hold, that's convergence, not sharp
    disagreement — stop rather than run another (costly) round."""
    return {state.bull.stance, state.bear.stance} == {"buy", "sell"}


def run_debate(
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
    news_text: str,
    base_url: str,
    max_rounds: int = 2,
    settings: GenSettings | None = None,
) -> tuple[DebateState, list[DebateState]]:
    """Returns (final_state, history) — history has one entry per round
    actually run, so callers can persist the full debate, not just the
    outcome (storage is free, unlike the LLM tokens spent producing it)."""
    from .llm_client import chat_completion

    bear_key_point = None
    state = None
    history: list[DebateState] = []
    for round_num in range(1, max_rounds + 1):
        bull_messages = build_debate_prompt(
            "bull",
            ticker,
            close,
            rsi_value,
            macd_hist,
            fired_reasons,
            news_text,
            bear_key_point,
        )
        bull_turn = parse_debate_response(
            chat_completion(bull_messages, base_url=base_url, settings=settings)
        )

        bear_messages = build_debate_prompt(
            "bear",
            ticker,
            close,
            rsi_value,
            macd_hist,
            fired_reasons,
            news_text,
            bull_turn.key_point,
        )
        bear_turn = parse_debate_response(
            chat_completion(bear_messages, base_url=base_url, settings=settings)
        )

        state = DebateState(round=round_num, bull=bull_turn, bear=bear_turn)
        history.append(state)
        bear_key_point = bear_turn.key_point

        if round_num == max_rounds or not should_continue_debate(state):
            break

    return state, history


_ACTIONS = {"buy", "hold", "sell"}

_TRADER_SYSTEM_PROMPT = """You are the Trader/Portfolio Manager. You are given \
the Bull and Bear analysts' final positions from a debate, plus the \
underlying real technical indicators — treat the indicators as ground truth. \
Make the final call. If action is Hold, use NA for the price/sizing fields.

Respond in EXACTLY this format, nothing else before or after it:

ACTION: <Buy, Hold, or Sell>
REASONING: <2-4 sentences>
ENTRY_PRICE: <a number, or NA>
STOP_LOSS: <a number, or NA>
POSITION_SIZING: <percent of portfolio as a number 0-100, or NA>"""


def build_trader_prompt(
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
    state: DebateState,
) -> list[dict]:
    context = format_market_context(ticker, close, rsi_value, macd_hist, fired_reasons)
    bull, bear = state.bull, state.bear
    user_prompt = (
        f"{context}\n"
        f"Bull: {bull.stance} ({bull.confidence}) — {bull.key_point}\n"
        f"Bear: {bear.stance} ({bear.confidence}) — {bear.key_point}"
    )
    return [
        {"role": "system", "content": _TRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class TraderDecision:
    action: str
    reasoning: str
    entry_price: float | None
    stop_loss: float | None
    position_sizing: float | None
    # Which sentinels fell back to a default — see SentimentSignal.fallbacks.
    fallbacks: frozenset[str] = frozenset()


_NA_VALUES = {"na", "n/a", "none", "-"}


def _extract_optional_float(text: str, field: str) -> tuple[float | None, bool]:
    """Returns (value, parsed_ok). Both a literal `NA` and a missing/garbage
    field yield None — the runtime contract, unchanged — but they are not the
    same event: `NA` is the correct answer on a Hold, whereas an absent or
    unparseable field is a format failure. parsed_ok separates them so eval can
    count only the failures."""
    raw = extract_field(text, field)
    if raw is not None and raw.strip().lower() in _NA_VALUES:
        return None, True
    try:
        return float(raw), True
    except (TypeError, ValueError):
        return None, False


def parse_trader_response(text: str) -> TraderDecision:
    fallbacks: set[str] = set()

    action = (extract_field(text, "ACTION") or "").lower()
    if action not in _ACTIONS:
        action = "hold"
        fallbacks.add("ACTION")

    reasoning = extract_field(text, "REASONING")
    if not reasoning:
        reasoning = "no reasoning parsed"
        fallbacks.add("REASONING")

    numbers = {}
    for field in ("ENTRY_PRICE", "STOP_LOSS", "POSITION_SIZING"):
        value, parsed_ok = _extract_optional_float(text, field)
        numbers[field] = value
        if not parsed_ok:
            fallbacks.add(field)

    return TraderDecision(
        action=action,
        reasoning=reasoning,
        entry_price=numbers["ENTRY_PRICE"],
        stop_loss=numbers["STOP_LOSS"],
        position_sizing=numbers["POSITION_SIZING"],
        fallbacks=frozenset(fallbacks),
    )


def run_trader(
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
    state: DebateState,
    base_url: str,
    settings: GenSettings | None = None,
) -> TraderDecision:
    from .llm_client import chat_completion

    messages = build_trader_prompt(
        ticker, close, rsi_value, macd_hist, fired_reasons, state
    )
    return parse_trader_response(
        chat_completion(messages, base_url=base_url, settings=settings)
    )
