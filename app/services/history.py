"""Helpers for the plain-text conversation history ("role: content" lines)."""

import re


_ROLE_LINE = re.compile(
    r"^\s*(user|assistant|system|bot|ai|human|customer|contact|agent|message)\s*:\s*",
    re.IGNORECASE,
)
_USER_ROLES = {"user", "human", "customer", "contact"}


def trim_history(history: str, limit: int) -> str:
    """Keep the most recent `limit` characters, cut at a line boundary."""

    history = (history or "").strip()

    if len(history) <= limit:
        return history

    tail = history[-limit:]
    newline = tail.find("\n")

    # Drop the partial first line unless the tail is one long line.
    if newline != -1:
        tail = tail[newline + 1:]

    return tail.strip()


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).casefold().strip(" .!?؟،")


def drop_current_message(history: str, message: str) -> str:
    """
    Remove the latest user message from the end of the history when the
    integration already included it in the transcript it sent.

    Without this, the current question is mistaken for the previous one,
    and the real previous topic is lost for follow-ups such as "كم سعره؟".
    """

    lines = (history or "").rstrip().splitlines()

    for index in range(len(lines) - 1, -1, -1):
        match = _ROLE_LINE.match(lines[index])

        if not match:
            continue

        if match.group(1).lower() not in _USER_ROLES:
            return history

        content = " ".join(
            [lines[index][match.end():]] + lines[index + 1:]
        )

        if _normalize(content) == _normalize(message):
            return "\n".join(lines[:index]).rstrip()

        return history

    return history
