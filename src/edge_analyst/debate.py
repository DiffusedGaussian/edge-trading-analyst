"""Phase 4 bull/bear debate loop. Carries a fixed-size DebateState between
rounds (overwritten each round, not appended to) rather than a growing
transcript — avoids TradingAgents' quadratic context-resend pattern while
keeping the round-to-round rebuttal fully intact.
"""

from __future__ import annotations

from dataclasses import dataclass

from .indicators import format_market_context
from .llm_parsing import extract_field

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


def parse_debate_response(text: str) -> DebateTurn:
    """Same forgiving line-scan pattern as analyst.py — hard defaults on any
    parse miss (Hold/low/placeholder), never raises."""
    stance = (extract_field(text, "STANCE") or "").lower()
    if stance not in _STANCES:
        stance = "hold"

    confidence = (extract_field(text, "CONFIDENCE") or "").lower()
    if confidence not in _CONFIDENCES:
        confidence = "low"

    key_point = extract_field(text, "KEY_POINT") or "no key point parsed"

    return DebateTurn(stance=stance, key_point=key_point, confidence=confidence)


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
) -> DebateState:
    from .llm_client import chat_completion

    bear_key_point = None
    state = None
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
            chat_completion(bull_messages, base_url=base_url)
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
            chat_completion(bear_messages, base_url=base_url)
        )

        state = DebateState(round=round_num, bull=bull_turn, bear=bear_turn)
        bear_key_point = bear_turn.key_point

        if round_num == max_rounds or not should_continue_debate(state):
            break

    return state


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


def _extract_optional_float(text: str, field: str) -> float | None:
    raw = extract_field(text, field)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None  # covers both a missing field and a literal "NA"


def parse_trader_response(text: str) -> TraderDecision:
    action = (extract_field(text, "ACTION") or "").lower()
    if action not in _ACTIONS:
        action = "hold"

    reasoning = extract_field(text, "REASONING") or "no reasoning parsed"

    return TraderDecision(
        action=action,
        reasoning=reasoning,
        entry_price=_extract_optional_float(text, "ENTRY_PRICE"),
        stop_loss=_extract_optional_float(text, "STOP_LOSS"),
        position_sizing=_extract_optional_float(text, "POSITION_SIZING"),
    )


def run_trader(
    ticker: str,
    close: float,
    rsi_value: float,
    macd_hist: float,
    fired_reasons: list[str],
    state: DebateState,
    base_url: str,
) -> TraderDecision:
    from .llm_client import chat_completion

    messages = build_trader_prompt(
        ticker, close, rsi_value, macd_hist, fired_reasons, state
    )
    return parse_trader_response(chat_completion(messages, base_url=base_url))
