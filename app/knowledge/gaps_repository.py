from sqlalchemy import text

from app.database.connection import engine


def insert_gap(
    client_id: int,
    message: str,
    gap_type: str,
):
    query = text("""
        INSERT INTO ai_gaps (
            client_id,
            message,
            type
        )
        VALUES (
            :client_id,
            :message,
            :gap_type
        )
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "client_id": client_id,
                "message": message,
                "gap_type": gap_type,
            },
        )