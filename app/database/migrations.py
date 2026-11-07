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
]


def run_migrations():
    with engine.begin() as connection:
        for statement in MIGRATIONS:
            connection.execute(text(statement))
