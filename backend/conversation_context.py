"""
Phase 21 — Lightweight conversation context management.

Keeps short-term conversation context useful without sending the entire
frontend history to the local Ollama model.
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_MAX_MESSAGES = 8
DEFAULT_MAX_MESSAGE_CHARS = 1200
DEFAULT_MAX_HISTORY_CHARS = 6000


_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:it|its|that|this|these|those|they|them|he|she|him|her)\b"
    r"|\b(?:what about|and what|also|then|where|when|why|how)\b",
    re.IGNORECASE,
)


_TOPIC_PATTERNS = (
    re.compile(r"\bwhat is\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\bwhat's\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\bwho created\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\btell me about\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\bexplain\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(
        r"\bhow does\s+(.+?)\s+work(?:[?.!]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\bdefine\s+(.+?)(?:[?.!]|$)", re.IGNORECASE),
)


def _message_parts(item: Any) -> tuple[str, str]:
    """Return normalized (role, content) for dict-like history entries."""

    if hasattr(item, "role") and hasattr(item, "content"):
        role = str(getattr(item, "role", "")).strip().lower()
        content = str(getattr(item, "content", "")).strip()
        return role, content

    if isinstance(item, dict):
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        return role, content

    return "", ""


def _trim_message(content: str, max_chars: int) -> str:
    """Trim long messages while keeping both the beginning and end."""

    if len(content) <= max_chars:
        return content

    marker = "\n...[message truncated]...\n"
    available = max(0, max_chars - len(marker))

    tail_chars = min(280, available // 4)
    head_chars = max(0, available - tail_chars)

    return (
        content[:head_chars].rstrip()
        + marker
        + content[-tail_chars:].lstrip()
    )[:max_chars]


def normalize_history(history: list[Any] | None) -> list[dict[str, str]]:
    """Keep only user/assistant messages with non-empty content."""

    if not history:
        return []

    normalized: list[dict[str, str]] = []

    for item in history:
        role, content = _message_parts(item)

        if role not in ("user", "assistant") or not content:
            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized


def compact_history(
    history: list[Any] | None,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    max_history_chars: int = DEFAULT_MAX_HISTORY_CHARS,
) -> list[dict[str, str]]:
    """
    Build a bounded recent history for Ollama.

    The newest messages are preferred. Individual messages and the total
    history are both capped so conversation growth cannot create a large
    prompt on the CPU-only setup.
    """

    max_messages = max(1, int(max_messages))
    max_message_chars = max(100, int(max_message_chars))
    max_history_chars = max(500, int(max_history_chars))

    normalized = normalize_history(history)
    recent = normalized[-max_messages:]

    compacted: list[dict[str, str]] = []
    used_chars = 0

    for item in reversed(recent):
        content = _trim_message(item["content"], max_message_chars)
        message_chars = len(content)

        if compacted and used_chars + message_chars > max_history_chars:
            break

        if not compacted and message_chars > max_history_chars:
            content = _trim_message(content, max_history_chars)
            message_chars = len(content)

        if used_chars + message_chars > max_history_chars:
            remaining = max_history_chars - used_chars

            if remaining < 100:
                break

            content = _trim_message(content, remaining)
            message_chars = len(content)

        compacted.append(
            {
                "role": item["role"],
                "content": content,
            }
        )
        used_chars += message_chars

    compacted.reverse()
    return compacted


def _clean_topic(topic: str) -> str:
    topic = re.sub(r"\s+", " ", topic).strip(" \\t\\r\\n?.!,;:")
    if len(topic) > 160:
        topic = topic[:160].rstrip() + "..."
    return topic


def extract_topic(text: str) -> str:
    """Extract a lightweight topic from a normal user question."""

    text = str(text or "").strip()

    if not text:
        return ""

    for pattern in _TOPIC_PATTERNS:
        match = pattern.search(text)

        if match:
            topic = _clean_topic(match.group(1))
            if topic:
                return topic

    return ""


def is_follow_up(text: str) -> bool:
    """Detect wording that commonly depends on a previous turn."""

    return bool(_FOLLOW_UP_PATTERN.search(str(text or "")))


def find_recent_topic(history: list[Any] | None) -> tuple[str, str]:
    """
    Return the newest useful topic and the user message it came from.

    Older turns are searched when the immediately previous user message was
    something short such as "yes" or "okay".
    """

    normalized = normalize_history(history)

    for item in reversed(normalized):
        if item["role"] != "user":
            continue

        topic = extract_topic(item["content"])
        if topic:
            return topic, item["content"]

    return "", ""


def build_reference_context(
    current_message: str,
    history: list[Any] | None,
) -> str:
    """
    Add a compact deterministic hint for pronoun/follow-up resolution.

    This does not rewrite the user's message or call another model.
    """

    if not is_follow_up(current_message):
        return ""

    topic, source_message = find_recent_topic(history)

    if not topic:
        return ""

    return (
        f"The current follow-up most likely refers to: {topic}.\n"
        f"Previous user topic: {source_message}"
    )


def history_for_prompt(
    history: list[Any] | None,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    max_history_chars: int = DEFAULT_MAX_HISTORY_CHARS,
) -> str:
    """Format compact history for the assistant prompt."""

    compacted = compact_history(
        history,
        max_messages=max_messages,
        max_message_chars=max_message_chars,
        max_history_chars=max_history_chars,
    )

    if not compacted:
        return ""

    lines = []

    for item in compacted:
        role_name = "USER" if item["role"] == "user" else "ASSISTANT"
        lines.append(f"{role_name}: {item['content']}")

    return "\n".join(lines)
