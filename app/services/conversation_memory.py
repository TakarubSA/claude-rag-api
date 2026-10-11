"""Short-lived in-process conversation memory for the chat API.

Memory is isolated by (client_id, conversation_id), held in RAM only,
and automatically expires after 30 minutes without activity. It is cleared
when the API process restarts.
"""

import threading
import time
from typing import Any

TTL_SECONDS = 30 * 60
CLEANUP_INTERVAL_SECONDS = 60
MAX_MESSAGES_PER_CONVERSATION = 20
MAX_CONTEXT_CHARS = 8000
MAX_CONVERSATIONS = 2000

_sessions: dict[tuple[int, str], dict[str, Any]] = {}
_lock = threading.Lock()


def _cleanup_expired_locked(now: float) -> None:
    expired_keys = [
        key
        for key, session in _sessions.items()
        if now - session["last_activity"] >= TTL_SECONDS
    ]
    for key in expired_keys:
        _sessions.pop(key, None)


def _trim_sessions_locked() -> None:
    """Bound RAM usage if an unusually large number of sessions appears."""
    excess = len(_sessions) - MAX_CONVERSATIONS
    if excess <= 0:
        return
    oldest = sorted(
        _sessions.items(),
        key=lambda item: item[1]["last_activity"],
    )[:excess]
    for key, _ in oldest:
        _sessions.pop(key, None)


def get_history(client_id: int, conversation_id: str) -> str:
    if not conversation_id:
        return ""

    key = (client_id, conversation_id)
    now = time.monotonic()
    with _lock:
        _cleanup_expired_locked(now)
        session = _sessions.get(key)
        if session is None:
            return ""
        session["last_activity"] = now
        messages = list(session["messages"])

    history = "\n".join(
        f'{item["role"]}: {item["content"]}' for item in messages
    )
    return history[-MAX_CONTEXT_CHARS:]


def save_turn(
    client_id: int,
    conversation_id: str | None,
    user_message: str,
    assistant_message: str,
) -> None:
    if not conversation_id:
        return

    user_message = (user_message or "").strip()
    assistant_message = (assistant_message or "").strip()
    if not user_message or not assistant_message:
        return

    key = (client_id, conversation_id)
    now = time.monotonic()
    with _lock:
        _cleanup_expired_locked(now)
        session = _sessions.setdefault(
            key,
            {"messages": [], "last_activity": now},
        )
        session["messages"].extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        session["messages"] = session["messages"][-MAX_MESSAGES_PER_CONVERSATION:]
        session["last_activity"] = now
        _trim_sessions_locked()


def clear_conversation(client_id: int, conversation_id: str | None) -> None:
    if not conversation_id:
        return
    with _lock:
        _sessions.pop((client_id, conversation_id), None)


def clear_all_conversations() -> None:
    """Clear all in-memory sessions; useful for tests or an explicit reset."""
    with _lock:
        _sessions.clear()


def _cleanup_worker() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        with _lock:
            _cleanup_expired_locked(time.monotonic())


threading.Thread(
    target=_cleanup_worker,
    name="conversation-memory-cleanup",
    daemon=True,
).start()
