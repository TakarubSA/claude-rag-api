from sqlalchemy import text

from app.database.connection import engine


def insert_knowledge(
    content: str,
    embedding: list[float],
    client_id: int,
):
    query = text("""
        INSERT INTO knowledge (
            content,
            embedding,
            client_id
        )
        VALUES (
            :content,
            :embedding,
            :client_id
        )
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "content": content,
                "embedding": embedding,
                "client_id": client_id,
            },
        )


def search_knowledge(
    query_embedding: list[float],
    client_id: int,
    limit: int = 3,
):


    query = text("""
        SELECT
            id,
            content,
            embedding <=> CAST(:embedding AS vector) AS distance
        FROM knowledge
        WHERE client_id = :client_id
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "embedding": str(query_embedding),
                "client_id": client_id,
                "limit": limit,
            },
        )

        return result.fetchall()


