from sqlalchemy import text

from app.database.connection import engine
import re


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
            CAST(:embedding AS vector),
            :client_id
        )
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "content": content,
                "embedding": str(embedding),
                "client_id": client_id,
            },
        )



_NORM = str.maketrans("إأآىة", "ااايه")
def _norm(s: str) -> str:
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s or "")
    return s.translate(_NORM)


def search_knowledge(
    query_embedding: list[float],
    client_id: int,
    limit: int = 8,
    query_text: str | None = None,
):
    """Hybrid retrieval (vector + keyword, fused with RRF), isolated by client_id."""
    tokens = re.findall(r"\w+", _norm(query_text))
    ts_query = " | ".join(tokens) or "''"

    query = text("""
        WITH sem AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:e AS vector)) AS rnk
            FROM knowledge
            WHERE client_id = :c
            ORDER BY embedding <=> CAST(:e AS vector)
            LIMIT :n
        ),
        lex AS (
            SELECT t.id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(t.d, q) DESC) AS rnk
            FROM (
                SELECT id,
                       to_tsvector('simple',
                           translate(
                               regexp_replace(content, '[\u064B-\u0652\u0640]', '', 'g'),
                               'إأآىة', 'ااايه')) AS d
                FROM knowledge
                WHERE client_id = :c
            ) t, to_tsquery('simple', :q) q
            WHERE t.d @@ q
            ORDER BY ts_rank_cd(t.d, q) DESC
            LIMIT :n
        )
        SELECT k.id, k.content,
               k.embedding <=> CAST(:e AS vector) AS distance,
               (l.id IS NOT NULL)::int AS keyword_hits,
               -(COALESCE(1.0/(60 + s.rnk), 0) + COALESCE(1.0/(60 + l.rnk), 0)) AS rank_score
        FROM sem s
        FULL OUTER JOIN lex l ON s.id = l.id
        JOIN knowledge k ON k.id = COALESCE(s.id, l.id)
        ORDER BY rank_score
        LIMIT :n
    """)

    with engine.connect() as connection:
        return connection.execute(
            query,
            {"e": str(query_embedding), "c": client_id, "n": limit, "q": ts_query},
        ).fetchall()
    """Semantic-only retrieval, isolated by client_id."""

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