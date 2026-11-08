from sqlalchemy import text

from app.database.connection import engine


# Idempotent: safe to run on every startup.
MIGRATIONS = [
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS anthropic_api_key text",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS credit_limit_usd numeric(12, 2)",
    # Admins do not belong to a client; providers do.
    "ALTER TABLE users ALTER COLUMN client_id DROP NOT NULL",
    "ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_user_role",
    """
    ALTER TABLE users ADD CONSTRAINT valid_user_role
        CHECK (role IN ('admin', 'provider', 'manager', 'viewer'))
    """,
    """
    CREATE TABLE IF NOT EXISTS claude_usage (
        id bigserial PRIMARY KEY,
        client_id integer NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
        model text NOT NULL,
        input_tokens integer NOT NULL,
        output_tokens integer NOT NULL,
        created_at timestamp NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS claude_usage_client_id_idx ON claude_usage (client_id)",
    "CREATE INDEX IF NOT EXISTS claude_usage_created_at_idx ON claude_usage (created_at)",
    # One row per customer message answered by /chat, for the dashboard.
    """
    CREATE TABLE IF NOT EXISTS chat_log (
        id bigserial PRIMARY KEY,
        client_id integer NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
        conversation_id text,
        message text NOT NULL,
        answer text NOT NULL DEFAULT '',
        action text NOT NULL,
        status text NOT NULL,
        outcome text NOT NULL,
        latency_ms integer,
        created_at timestamp NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS chat_log_client_created_idx ON chat_log (client_id, created_at)",
    # ai_gaps is not touched here: in production it belongs to another
    # database role, and ALTER TABLE on it fails even when nothing changes.
    # The dashboard reads and writes these two columns.
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS created_at timestamp NOT NULL DEFAULT now()",
    "ALTER TABLE clients ADD COLUMN IF NOT EXISTS updated_at timestamp NOT NULL DEFAULT now()",
    # Cost per message: each Claude call is linked to the chat request that
    # made it, labelled with its pipeline step, and priced when recorded.
    "ALTER TABLE claude_usage ADD COLUMN IF NOT EXISTS request_id uuid",
    "ALTER TABLE claude_usage ADD COLUMN IF NOT EXISTS purpose text",
    "ALTER TABLE claude_usage ADD COLUMN IF NOT EXISTS cost_usd numeric(12, 6)",
    "CREATE INDEX IF NOT EXISTS claude_usage_request_id_idx ON claude_usage (request_id)",
    "ALTER TABLE chat_log ADD COLUMN IF NOT EXISTS request_id uuid",
    "ALTER TABLE chat_log ADD COLUMN IF NOT EXISTS intent text",
    "ALTER TABLE chat_log ADD COLUMN IF NOT EXISTS intent_reason text",
    "ALTER TABLE chat_log ADD COLUMN IF NOT EXISTS best_distance real",
    "CREATE UNIQUE INDEX IF NOT EXISTS chat_log_request_id_idx ON chat_log (request_id)",
]

# Rows recorded before cost_usd existed are priced at the current prices.
BACKFILL_COST = """
    UPDATE claude_usage
    SET cost_usd = round(
        input_tokens / 1000000.0 * CAST(:input_price AS numeric)
        + output_tokens / 1000000.0 * CAST(:output_price AS numeric),
        6
    )
    WHERE cost_usd IS NULL
"""


def run_migrations():
    from app.llm.usage import INPUT_PRICE_PER_MTOK, OUTPUT_PRICE_PER_MTOK

    with engine.begin() as connection:
        for statement in MIGRATIONS:
            connection.execute(text(statement))

        connection.execute(text(BACKFILL_COST), {
            "input_price": INPUT_PRICE_PER_MTOK,
            "output_price": OUTPUT_PRICE_PER_MTOK,
        })
