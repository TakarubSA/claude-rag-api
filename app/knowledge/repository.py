import re

from sqlalchemy import text

from app.database.connection import engine


_NORM = str.maketrans("إأآىة", "ااايه")

_STOP = {
    "هل", "ما", "ماذا", "كيف", "وين", "اين", "متي", "من", "في", "علي", "عن", "الي",
    "ان", "او", "كم", "هو", "هي", "مع", "لا", "نعم", "عندكم", "عندك", "عندي",
    "ابي", "ابغي", "اقدر", "يمكنني", "يمكن", "ممكن", "تقدمون", "لدي", "انا",
    "هذا", "هذه", "فيه", "ليش", "وش", "ايش", "شو",
}


def _norm(s: str) -> str:
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s or "")
    return s.translate(_NORM)


def _tokens(s: str) -> list[str]:
    out = []
    for t in re.findall(r"\w+", _norm(s)):
        if t.startswith("ال") and len(t) > 4:
            t = t[2:]  # "التامين" -> "تامين"
        if len(t) > 2 and t not in _STOP:
            out.append(t)
    return out


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


def search_knowledge(
    query_embedding: list[float],
    client_id: int,
    limit: int = 8,
    query_text: str | None = None,
):
    """Hybrid retrieval (vector + keyword, fused with RRF), isolated by client_id."""
    ts_query = " | ".join(_tokens(query_text)) or "''"

    query = text(r"""
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
                           regexp_replace(
                               translate(
                                   regexp_replace(content, '[\u064B-\u0652\u0640]', '', 'g'),
                                   'إأآىة', 'ااايه'),
                               '\mال(?=\w{3})', '', 'g')) AS d
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