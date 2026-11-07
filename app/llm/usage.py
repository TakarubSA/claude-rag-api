import logging
import os
from contextvars import ContextVar
from functools import lru_cache

import anthropic
from dotenv import load_dotenv
from sqlalchemy import text

from app.database.connection import engine


load_dotenv()

logger = logging.getLogger(__name__)


# USD per million tokens, used only for the dashboard's cost estimate.
# Defaults are Claude Haiku 4.5 list prices; override if the model changes.
INPUT_PRICE_PER_MTOK = float(os.getenv("CLAUDE_INPUT_PRICE_PER_MTOK", "1"))
OUTPUT_PRICE_PER_MTOK = float(os.getenv("CLAUDE_OUTPUT_PRICE_PER_MTOK", "5"))

DEFAULT_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# Which provider the current request is for: (client_id, api_key).
# Set once per request by the chat route; every Claude call on that
# thread then uses the provider's own key and is billed to the provider.
_current_client: ContextVar[tuple[int | None, str | None]] = ContextVar(
    "current_claude_client",
    default=(None, None),
)


INSERT_USAGE_QUERY = text("""
    INSERT INTO claude_usage (client_id, model, input_tokens, output_tokens)
    VALUES (:client_id, :model, :input_tokens, :output_tokens)
""")


def use_client(client_id: int, api_key: str | None):
    _current_client.set((client_id, api_key or None))


# A normal call takes 1-2 s. The SDK default waits up to 10 minutes on a
# stuck request; give up after 15 s instead and let the SDK retry.
REQUEST_TIMEOUT_SECONDS = 15


@lru_cache(maxsize=128)
def _anthropic(api_key: str | None) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=2,
    )


def create_message(**kwargs):
    """messages.create with the current provider's key, recording its usage."""
    client_id, api_key = _current_client.get()

    response = _anthropic(api_key or DEFAULT_API_KEY).messages.create(**kwargs)

    if client_id is not None:
        _record_usage(client_id, response)

    return response


def _record_usage(client_id: int, response):
    usage = getattr(response, "usage", None)

    if usage is None:
        return

    # Usage is reporting only: never fail the user's answer over it.
    try:
        with engine.begin() as connection:
            connection.execute(INSERT_USAGE_QUERY, {
                "client_id": client_id,
                "model": response.model,
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
            })
    except Exception:
        logger.exception("Failed to record Claude usage: client_id=%s", client_id)


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK,
        4,
    )
