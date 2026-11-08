import secrets

from sqlalchemy import text

from app.database.connection import engine


PROVIDER_COLUMNS = """
    c.id,
    c.name,
    c.token,
    c.anthropic_api_key,
    c.credit_limit_usd,
    c.created_at,
    (
        SELECT email FROM users
        WHERE client_id = c.id AND role = 'provider'
        ORDER BY id LIMIT 1
    ) AS email,
    (SELECT count(*) FROM knowledge WHERE client_id = c.id) AS knowledge_count,
    COALESCE(u.calls, 0) AS calls,
    COALESCE(u.input_tokens, 0) AS input_tokens,
    COALESCE(u.output_tokens, 0) AS output_tokens,
    COALESCE(u.cost_usd, 0) AS cost_usd
"""

USAGE_JOIN = """
    LEFT JOIN (
        SELECT client_id,
               count(*) AS calls,
               sum(input_tokens) AS input_tokens,
               sum(output_tokens) AS output_tokens,
               sum(cost_usd) AS cost_usd
        FROM claude_usage
        GROUP BY client_id
    ) u ON u.client_id = c.id
"""


def _mask_key(key: str | None) -> str | None:
    return f"…{key[-4:]}" if key else None


def _provider_dict(row) -> dict:
    cost = round(float(row.cost_usd), 4)
    limit = float(row.credit_limit_usd) if row.credit_limit_usd is not None else None

    return {
        "id": row.id,
        "name": row.name,
        "email": row.email,
        "token": row.token,
        # The full key is never sent back to the browser.
        "api_key_hint": _mask_key(row.anthropic_api_key),
        "credit_limit_usd": limit,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "knowledge_count": row.knowledge_count,
        "usage": {
            "calls": row.calls,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "cost_usd": cost,
            "remaining_usd": round(limit - cost, 2) if limit is not None else None,
        },
    }


def get_user_by_email(email: str):
    with engine.connect() as connection:
        return connection.execute(
            text("""
                SELECT id, client_id, name, email, password_hash, role
                FROM users
                WHERE lower(email) = lower(:email)
            """),
            {"email": email},
        ).fetchone()


def email_exists(email: str) -> bool:
    return get_user_by_email(email) is not None


def list_providers() -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(text(f"""
            SELECT {PROVIDER_COLUMNS}
            FROM clients c
            {USAGE_JOIN}
            ORDER BY c.id
        """)).fetchall()

    return [_provider_dict(row) for row in rows]


def get_provider(client_id: int) -> dict | None:
    with engine.connect() as connection:
        row = connection.execute(
            text(f"""
                SELECT {PROVIDER_COLUMNS}
                FROM clients c
                {USAGE_JOIN}
                WHERE c.id = :id
            """),
            {"id": client_id},
        ).fetchone()

    return _provider_dict(row) if row else None


def create_provider(
    name: str,
    email: str,
    password_hash: str,
    anthropic_api_key: str | None,
    credit_limit_usd: float | None,
) -> int:
    """Create the client and its provider login together, in one transaction."""
    with engine.begin() as connection:
        client_id = connection.execute(
            text("""
                INSERT INTO clients (name, token, anthropic_api_key, credit_limit_usd)
                VALUES (:name, :token, :key, :limit)
                RETURNING id
            """),
            {
                "name": name,
                "token": secrets.token_urlsafe(32),
                "key": anthropic_api_key or None,
                "limit": credit_limit_usd,
            },
        ).scalar_one()

        connection.execute(
            text("""
                INSERT INTO users (client_id, name, email, password_hash, role)
                VALUES (:client_id, :name, :email, :password_hash, 'provider')
            """),
            {
                "client_id": client_id,
                "name": name,
                "email": email,
                "password_hash": password_hash,
            },
        )

    return client_id


def add_provider_login(
    client_id: int,
    name: str,
    email: str,
    password_hash: str,
    anthropic_api_key: str | None,
    credit_limit_usd: float | None,
):
    """Give an existing client a provider login. Key and limit change only when given."""
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO users (client_id, name, email, password_hash, role)
                VALUES (:client_id, :name, :email, :password_hash, 'provider')
            """),
            {
                "client_id": client_id,
                "name": name,
                "email": email,
                "password_hash": password_hash,
            },
        )

        if anthropic_api_key:
            connection.execute(
                text("UPDATE clients SET anthropic_api_key = :key WHERE id = :id"),
                {"key": anthropic_api_key, "id": client_id},
            )

        if credit_limit_usd is not None:
            connection.execute(
                text("UPDATE clients SET credit_limit_usd = :limit WHERE id = :id"),
                {"limit": credit_limit_usd, "id": client_id},
            )


def update_provider(client_id: int, changes: dict):
    allowed = {"anthropic_api_key", "credit_limit_usd"}
    changes = {k: v for k, v in changes.items() if k in allowed}

    if not changes:
        return

    assignments = ", ".join(f"{column} = :{column}" for column in changes)

    with engine.begin() as connection:
        connection.execute(
            text(f"UPDATE clients SET {assignments}, updated_at = now() WHERE id = :id"),
            {**changes, "id": client_id},
        )


def regenerate_token(client_id: int):
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE clients SET token = :token, updated_at = now() WHERE id = :id"),
            {"token": secrets.token_urlsafe(32), "id": client_id},
        )
