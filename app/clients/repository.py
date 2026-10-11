import logging

from sqlalchemy import text

from app.database.connection import engine


logger = logging.getLogger(__name__)


GET_CLIENT_BY_TOKEN_QUERY = text("""
    SELECT
        id,
        name,
        ai_config
    FROM clients
    WHERE token = :token
    LIMIT 1
""")


def get_client_by_token(token: str):
    """
    Retrieve a client using its API token.

    Returns:
        The SQLAlchemy Row if the client exists, otherwise None.
    """
    if not token or not token.strip():
        logger.warning("Client lookup attempted with an empty token.")
        return None

    with engine.connect() as connection:
        client = connection.execute(
            GET_CLIENT_BY_TOKEN_QUERY,
            {"token": token},
        ).fetchone()

    # Never log the token or the complete client/configuration object.
    logger.debug(
        "Client token lookup: %s",
        "matched" if client is not None else "no match",
    )

    return client