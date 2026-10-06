from sqlalchemy import text

from app.database.connection import engine


def get_client_by_token(token: str):
    query = text("""
        SELECT id, name
        FROM clients
        WHERE token = :token
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "token": token,
            },
        )

        client = result.fetchone()

        print("TOKEN RECEIVED:", repr(token))
        print("CLIENT FOUND:", client)

        return client