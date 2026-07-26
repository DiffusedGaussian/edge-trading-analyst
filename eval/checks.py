"""Tier 0: deterministic checks over one model response. No I/O, no network,
no LLM, no judge — every function here is a pure function of already-extracted
strings, so a failure is attributable to the model under test and nothing else.

These are the checks that can be *wrong* about the model only in the way a
compiler can be wrong: mechanically. Everything requiring taste belongs to
Tier 1 (eval/rubric.py). The division matters because a Tier 1 `no` verdict
can't be attributed — it could be the cascade or the judge — whereas a Tier 0
failure is a fact about the output.

Design rule throughout: prefer a false negative to a false positive. A check
that fires spuriously trains you to ignore the scorecard, which is worse than a
check that occasionally misses. Hence the negation handling in
check_label_consistency and the deliberate skips (rather than failures) when an
input can't support a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    # Human-readable why, always populated — even on a pass, so a scorecard
    # line is self-explanatory without re-running anything.
    detail: str


# --- 1. fields parsed --------------------------------------------------------


def check_fields_parsed(fallbacks: frozenset[str]) -> CheckResult:
    """The single most diagnostic check: did every sentinel field actually
    parse? A model whose output is unparseable scores as neutral/5.0/low at
    runtime, which reads as mediocre rather than broken. This separates them."""
    if not fallbacks:
        return CheckResult("fields_parsed", True, "all sentinel fields parsed")
    return CheckResult(
        "fields_parsed",
        False,
        f"fell back to defaults for: {', '.join(sorted(fallbacks))}",
    )


# --- 2. truncation -----------------------------------------------------------


def check_not_truncated(finish_reason: str) -> CheckResult:
    """`length` means the response was cut off at max_tokens. The forgiving
    parser turns that into fallbacks with no hint of the cause — a long
    thinking trace looks identical to a model that can't follow the format."""
    if finish_reason == "length":
        return CheckResult(
            "not_truncated", False, "hit max_tokens (finish_reason=length)"
        )
    return CheckResult("not_truncated", True, f"finish_reason={finish_reason}")


# --- 3. label consistency ----------------------------------------------------

# format_market_context already hands the model the deterministic label next to
# the raw number, so contradicting it is not a difference of opinion — it is a
# mechanical failure. This is the check that catches the 2026-07-23 gemma-3-1b
# regression: RSI 61.7 described as "strong bullish momentum" against 30/70.
_RSI_FORBIDDEN: dict[str, frozenset[str]] = {
    "neutral": frozenset({"overbought", "oversold"}),
    "overbought": frozenset({"oversold"}),
    "oversold": frozenset({"overbought"}),
}

_BEARISH_PHRASES = frozenset(
    {
        "bearish momentum",
        "negative momentum",
        "momentum is fading",
        "momentum is negative",
    }
)
_BULLISH_PHRASES = frozenset(
    {
        "bullish momentum",
        "positive momentum",
        "momentum is building",
        "momentum is positive",
    }
)

_MACD_FORBIDDEN: dict[str, frozenset[str]] = {
    "bullish momentum": _BEARISH_PHRASES,
    "bearish momentum": _BULLISH_PHRASES,
    # A zero histogram supports neither direction.
    "flat": _BEARISH_PHRASES | _BULLISH_PHRASES,
}

# Without these, the check false-positives on correct output constantly: "RSI is
# not overbought" contains "overbought". Scanned over the 20 characters
# preceding a match, which is enough for the usual "is not X" / "rather than X"
# constructions without reaching back into an unrelated clause.
NEGATION_MARKERS: tuple[str, ...] = (
    "not ",
    "n't",
    "isn't",
    "rather than",
    "far from",
    "nowhere near",
)
_NEGATION_WINDOW = 20


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - _NEGATION_WINDOW) : start]
    return any(marker in window for marker in NEGATION_MARKERS)


def _forbidden_hits(free_text: str, forbidden: frozenset[str]) -> list[str]:
    lowered = free_text.lower()
    hits = []
    for term in sorted(forbidden):
        for match in re.finditer(re.escape(term), lowered):
            if not _is_negated(lowered, match.start()):
                hits.append(term)
                break
    return hits


def check_label_consistency(
    free_text: str, rsi_label: str, macd_label: str
) -> CheckResult:
    """Scans the model's own prose for terms that contradict the deterministic
    indicator labels it was given.

    `free_text` must be the concatenated parsed values (RATIONALE / KEY_POINT /
    REASONING) and *never* the raw response — some models echo the prompt's
    indicator block in a preamble, and those labels are the prompt's, not the
    model's claim about the market.
    """
    hits = _forbidden_hits(free_text, _RSI_FORBIDDEN.get(rsi_label, frozenset()))
    hits += _forbidden_hits(free_text, _MACD_FORBIDDEN.get(macd_label, frozenset()))
    if hits:
        return CheckResult(
            "label_consistency",
            False,
            f"contradicts RSI={rsi_label} / MACD={macd_label}: "
            f"said {', '.join(repr(h) for h in hits)}",
        )
    return CheckResult(
        "label_consistency",
        True,
        f"no terms contradicting RSI={rsi_label} / MACD={macd_label}",
    )


# --- 4. numeric fidelity -----------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

# Numbers a model may legitimately cite without them appearing in the input:
# the RSI band values it was told about implicitly, and the 0-10 score scale.
ALLOWED_BARE_NUMBERS = frozenset({"0", "30", "50", "70", "100"})


def _normalise_number(raw: str) -> str:
    """`12.0%` -> `12`, `0.400` -> `0.4`, so a model restating an input number
    with different trailing zeros isn't flagged as inventing one."""
    value = raw.rstrip("%")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def check_numeric_fidelity(free_text: str, prompt_text: str) -> CheckResult:
    """Every number in the model's prose must be traceable to its input.

    A small model inventing a plausible-looking figure ("revenue up 15%" when
    the news said 12%) is the failure mode here, and it is invisible to a judge
    that doesn't have the input side by side.

    `free_text` must be prose only. The sentinel *values* (SCORE: 7,
    ENTRY_PRICE: 150.0) are numbers the model is supposed to generate, so
    passing the raw response here would flag correct output.
    """
    prompt_numbers = {
        _normalise_number(m.group()) for m in _NUMBER_RE.finditer(prompt_text)
    }
    unsupported = []
    for match in _NUMBER_RE.finditer(free_text):
        raw = match.group()
        value = _normalise_number(raw)
        if value in prompt_numbers or value in ALLOWED_BARE_NUMBERS:
            continue
        # An integer on the 0-10 score scale is also fair game.
        if value.isdigit() and 0 <= int(value) <= 10:
            continue
        # Last resort: the number may appear in the prompt in a form the
        # tokeniser above split differently (inside a longer figure).
        if value in prompt_text:
            continue
        unsupported.append(raw)

    if unsupported:
        return CheckResult(
            "numeric_fidelity",
            False,
            f"numbers not traceable to the input: {', '.join(unsupported)}",
        )
    return CheckResult(
        "numeric_fidelity", True, "every number in the prose traces to the input"
    )


# --- 5. news grounding -------------------------------------------------------

_TOKEN_RE = re.compile(r"\W+")
_MIN_TOKEN_LEN = 4

_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "there",
        "their",
        "them",
        "then",
        "than",
        "when",
        "what",
        "which",
        "while",
        "into",
        "over",
        "under",
        "about",
        "after",
        "before",
        "more",
        "most",
        "some",
        "such",
        "only",
        "also",
        "very",
        "much",
        "both",
        "each",
        "other",
        "against",
        "because",
        "however",
        "given",
        "being",
        "does",
        "doing",
        "here",
        "they",
        "these",
        "those",
        "your",
        "ours",
    }
)

_NO_NEWS_VALUES = frozenset({"", "none", "n/a", "na", "no news"})


def _is_no_news(news_text: str | None) -> bool:
    return (news_text or "").strip().lower() in _NO_NEWS_VALUES


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.split((text or "").lower())
        if len(token) >= _MIN_TOKEN_LEN and token not in _STOPWORDS
    }


def check_news_grounding(
    rationale: str, news_text: str, prompt_boilerplate: str
) -> CheckResult:
    """The prompt demands the rationale reference a specific fact from the
    input; nothing verified that it did.

    Tokens appearing in the system prompt are subtracted, so generic domain
    vocabulary the prompt itself supplied — `indicator`, `bullish`, `momentum`,
    `technical` — cannot pass as evidence of having read the news.
    """
    if _is_no_news(news_text):
        return CheckResult("news_grounding", True, "no news to ground in (skipped)")

    boilerplate = _content_tokens(prompt_boilerplate)
    news_tokens = _content_tokens(news_text) - boilerplate
    if not news_tokens:
        # Every distinctive word in the news also appears in the system prompt,
        # so no token could ever prove grounding. Skipping is the honest
        # verdict; failing here would penalise the model for our prompt.
        return CheckResult(
            "news_grounding", True, "news has no vocabulary distinct from the prompt"
        )

    shared = (_content_tokens(rationale) - boilerplate) & news_tokens
    if shared:
        return CheckResult(
            "news_grounding",
            True,
            f"rationale shares {', '.join(sorted(shared))} with the news",
        )
    return CheckResult(
        "news_grounding",
        False,
        "rationale shares no content word with the news "
        f"(news offered: {', '.join(sorted(news_tokens)[:8])})",
    )


# --- 6. fabricated news ------------------------------------------------------

# Deliberately crude. It is not trying to detect subtle misrepresentation —
# only a model inventing a story where there was no story at all.
EVENT_VERBS = frozenset(
    {
        "announced",
        "reported",
        "launched",
        "filed",
        "beat",
        "missed",
        "earnings",
        "acquisition",
        "guidance",
        "recall",
    }
)


def check_no_fabricated_news(rationale: str, news_text: str) -> CheckResult:
    """Applies only when there was no news. With news present, deciding whether
    a claim misrepresents it needs judgment — that's Tier 1's job."""
    if not _is_no_news(news_text):
        return CheckResult(
            "no_fabricated_news", True, "news was present (skipped, Tier 1 covers it)"
        )
    found = sorted(EVENT_VERBS & _content_tokens(rationale))
    if found:
        return CheckResult(
            "no_fabricated_news",
            False,
            f"claims an event with no news in the input: {', '.join(found)}",
        )
    return CheckResult("no_fabricated_news", True, "no event claimed without news")


# --- 7. degeneracy -----------------------------------------------------------

_MAX_LINE_REPEATS = 3
_ECHO_WINDOW = 40


def check_not_degenerate(raw_output: str, prompt_text: str) -> CheckResult:
    """Catches the three ways a small model fails to produce anything at all:
    silence, a repetition loop, and parroting its own prompt back."""
    if not raw_output or not raw_output.strip():
        return CheckResult("not_degenerate", False, "output is empty or whitespace")

    counts: dict[str, int] = {}
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = [line for line, n in counts.items() if n >= _MAX_LINE_REPEATS]
    if repeated:
        worst = max(repeated, key=lambda line: counts[line])
        return CheckResult(
            "not_degenerate",
            False,
            f"line repeated {counts[worst]}x: {worst[:60]!r}",
        )

    # Prompt echo: any long verbatim span of the prompt reappearing in the
    # output. 40 chars is long enough that a shared phrase is not coincidence.
    for start in range(0, max(0, len(raw_output) - _ECHO_WINDOW) + 1):
        window = raw_output[start : start + _ECHO_WINDOW]
        if len(window) == _ECHO_WINDOW and window in prompt_text:
            return CheckResult(
                "not_degenerate", False, f"echoes the prompt verbatim: {window!r}"
            )

    return CheckResult("not_degenerate", True, "non-empty, no repetition or echo")


# --- 8. sentinel hijacking ---------------------------------------------------

SENTINEL_NAMES: tuple[str, ...] = (
    "LABEL",
    "SCORE",
    "CONFIDENCE",
    "RATIONALE",
    "STANCE",
    "KEY_POINT",
    "ACTION",
    "REASONING",
    "ENTRY_PRICE",
    "STOP_LOSS",
    "POSITION_SIZING",
)

_INJECTED_SENTINEL_RE = re.compile(
    rf"^\s*({'|'.join(SENTINEL_NAMES)}):\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def check_sentinel_not_hijacked(
    raw_output: str, news_text: str, parsed_values: dict[str, str]
) -> CheckResult:
    """extract_field does a MULTILINE scan of the response, and news_text is
    embedded in the prompt — so a headline containing a sentinel line that the
    model echoes can drive the parsed value. That is a prompt injection with a
    real consequence: a decision the model never made.

    Fails only when an injected value actually *became* the parsed value; a
    model that ignores the injection passes.
    """
    injections = [
        (m.group(1).upper(), m.group(2).strip())
        for m in _INJECTED_SENTINEL_RE.finditer(news_text or "")
    ]
    if not injections:
        return CheckResult(
            "sentinel_not_hijacked", True, "no sentinel line in the news (skipped)"
        )

    hijacked = [
        f"{name}={value!r}"
        for name, value in injections
        if str(parsed_values.get(name, "")).strip().lower() == value.lower()
    ]
    if hijacked:
        return CheckResult(
            "sentinel_not_hijacked",
            False,
            f"parsed value came from the injected news line: {', '.join(hijacked)}",
        )
    return CheckResult(
        "sentinel_not_hijacked",
        True,
        f"resisted {len(injections)} injected sentinel line(s)",
    )


# --- orchestration -----------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CheckInputs:
    """Everything the checks need, bundled so run_all_checks doesn't take ten
    positional strings that are trivially easy to transpose.

    free_text is the concatenation of the model's *parsed* prose values
    (RATIONALE / KEY_POINT / REASONING) — deliberately not raw_output, see
    check_label_consistency. prompt_text is the full rendered prompt (system +
    user); prompt_boilerplate is the system message alone.
    """

    raw_output: str
    prompt_text: str
    prompt_boilerplate: str
    free_text: str
    rsi_label: str
    macd_label: str
    news_text: str
    fallbacks: frozenset[str] = frozenset()
    finish_reason: str = "unknown"
    parsed_values: dict[str, str] = field(default_factory=dict)


# Fixed order so two scorecards diff cleanly, regardless of dict iteration or
# which stage produced them.
CHECK_ORDER: tuple[str, ...] = (
    "fields_parsed",
    "not_truncated",
    "not_degenerate",
    "label_consistency",
    "numeric_fidelity",
    "news_grounding",
    "no_fabricated_news",
    "sentinel_not_hijacked",
)


def run_all_checks(inputs: CheckInputs) -> list[CheckResult]:
    results = [
        check_fields_parsed(inputs.fallbacks),
        check_not_truncated(inputs.finish_reason),
        check_not_degenerate(inputs.raw_output, inputs.prompt_text),
        check_label_consistency(inputs.free_text, inputs.rsi_label, inputs.macd_label),
        check_numeric_fidelity(inputs.free_text, inputs.prompt_text),
        check_news_grounding(
            inputs.free_text, inputs.news_text, inputs.prompt_boilerplate
        ),
        check_no_fabricated_news(inputs.free_text, inputs.news_text),
        check_sentinel_not_hijacked(
            inputs.raw_output, inputs.news_text, inputs.parsed_values
        ),
    ]
    by_name = {r.name: r for r in results}
    return [by_name[name] for name in CHECK_ORDER]
