"""Shared forgiving-parser primitive for sentinel key-value LLM output.
Used by both news_analyst.py and debate.py — same failure modes, same fix."""

from __future__ import annotations

import re


def extract_field(text: str, field: str) -> str | None:
    """Finds `FIELD: value` anywhere in text (tolerant of leading whitespace,
    case, and surrounding prose/markdown), returns None if absent."""
    match = re.search(rf"^\s*{field}:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None
