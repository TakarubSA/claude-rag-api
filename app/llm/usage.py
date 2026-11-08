import logging
import os
import uuid
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


# Which provider and request the current work is for:
# (client_id, api_key, request_id). Set once per request by the chat route;
# every Claude call on that thread then uses the provider's own key, is
# billed to the provider and is linked to the request's chat_log row.
_current_client: ContextVar[tuple[int | None, str | None, str | None]] = ContextVar(
    "current_claude_client",
    default=(None, None, None),
)


INSERT_USAGE_QUERY = text("""
    INSERT INTO claude_usage (
        client_id, request_id, purpose, model,
        input_tokens, output_tokens, cost_usd
    )
    VALUES (
        :client_id, :request_id, :purpose, :model,
        :input_tokens, :output_tokens, :cost_usd
    )
""")


def use_client(client_id: int, api_key: str | None) -> str:
    """Start a request for this provider. Returns the new request id."""
    request_id = str(uuid.uuid4())
    _current_client.set((client_id, api_key or None, request_id))
    return request_id


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


def create_message(purpose: str | None = None, **kwargs):
    """
    messages.create with the current provider's key, recording its usage.

    `purpose` names the pipeline step (intent, answer, ...) for the
    dashboard's cost breakdown.
    """
    client_id, api_key, request_id = _current_client.get()

    response = _anthropic(api_key or DEFAULT_API_KEY).messages.create(**kwargs)

    if client_id is not None:
        _record_usage(client_id, request_id, purpose, response)

    return response


def _record_usage(client_id: int, request_id: str | None, purpose: str | None, response):
    usage = getattr(response, "usage", None)

    if usage is None:
        return

    # Usage is reporting only: never fail the user's answer over it.
    try:
        with engine.begin() as connection:
            input_tokens = usage.input_tokens or 0
            output_tokens = usage.output_tokens or 0
            connection.execute(INSERT_USAGE_QUERY, {
                "client_id": client_id,
                "request_id": request_id,
                "purpose": purpose,
                "model": response.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # Stored at call time, so later price changes do not
                # rewrite past spending.
                "cost_usd": estimate_cost(input_tokens, output_tokens, precision=6),
            })
    except Exception:
        logger.exception("Failed to record Claude usage: client_id=%s", client_id)


def estimate_cost(input_tokens: int, output_tokens: int, precision: int = 4) -> float:
    return round(
        input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK,
        precision,
    )
