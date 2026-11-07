import re

from sqlalchemy import text

from app.database.connection import engine


_NORM = str.maketrans("إأآىة", "ااايه")

# Generic question and intent words that should not dominate lexical search.
# This prevents broad wording from matching unrelated knowledge entries.
_STOP = {
    "هل", "ما", "ماذا", "كيف", "وين", "اين", "متي", "من", "في", "علي", "عن", "الي",
    "ان", "او", "كم", "هو", "هي", "مع", "لا", "نعم", "عندكم", "عندك", "عندي",
    "ابي", "ابغي", "اقدر", "يمكنني", "يمكن", "ممكن", "تقدمون", "لدي", "انا",
    "هذا", "هذه", "فيه", "ليش", "وش", "ايش", "شو",
    "سعر", "السعر", "اسعار", "الاسعار", "تكلفه", "تكلفة", "يكلف", "بكم", "بكام",
    "كمية", "الكمية", "متوفر", "متوفره", "متوفرين", "عند", "لو", "سمحت",
}


def _norm(s: str) -> str:
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s or "")
    return s.translate(_NORM)


def _tokens(s: str) -> list[str]:
    out = []
    for t in re.findall(r"\w+", _norm(s)):
        if t.startswith("ال") and len(t) > 4:
            t = t[2:]
        if len(t) > 2 and t not in _STOP:
            out.append(t)
    # Keep order while removing duplicate tokens.
    return list(dict.fromkeys(out))


INSERT_KNOWLEDGE_QUERY = text("""
    INSERT INTO knowledge (content, embedding, client_id)
    VALUES (:content, CAST(:embedding AS vector), :client_id)
""")


def insert_knowledge(content: str, embedding: list[float], client_id: int):
    insert_knowledge_many([content], [embedding], client_id)


def insert_knowledge_many(
    contents: list[str],
    embeddings: list[list[float]],
    client_id: int,
):
    """Insert all entries in one transaction: either every row is saved or none."""
    if len(contents) != len(embeddings):
        raise ValueError(
            f"Got {len(contents)} entries but {len(embeddings)} embeddings"
        )

    if not contents:
        return

    with engine.begin() as connection:
        connection.execute(INSERT_KNOWLEDGE_QUERY, [
            {
                "content": content,
                "embedding": str(embedding),
                "client_id": client_id,
            }
            for content, embedding in zip(contents, embeddings)
        ])


def search_knowledge(
    query_embedding: list[float],
    client_id: int,
    limit: int = 8,
    query_text: str | None = None,
):
    """Tenant-isolated hybrid retrieval; lexical matching focuses on subject terms."""
    lexical_tokens = _tokens(query_text or "")
    # Never let a generic word such as "سعر" retrieve unrelated prices on its own.
    # Semantic retrieval still uses the complete question embedding.
    ts_query = " | ".join(lexical_tokens) or "zzzzlexicalnomatchzzzz"

    query = text(r"""
        WITH sem AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:e AS vector)) AS rnk
            FROM knowledge
            WHERE client_id = :c
            ORDER BY embedding <=> CAST(:e AS vector)
            LIMIT :n
        ), lex AS (
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
            ) t
            CROSS JOIN to_tsquery('simple', :q) q
            WHERE t.d @@ q
            ORDER BY ts_rank_cd(t.d, q) DESC
            LIMIT :n
        )
        SELECT k.id, k.content,
               k.embedding <=> CAST(:e AS vector) AS distance,
               (l.id IS NOT NULL)::int AS keyword_hits,
               -(CASE WHEN l.id IS NOT NULL THEN 1.0 ELSE 0.0 END +
                 COALESCE(1.0/(60 + s.rnk), 0) +
                 COALESCE(1.0/(60 + l.rnk), 0)) AS rank_score
        FROM sem s
        FULL OUTER JOIN lex l ON s.id = l.id
        JOIN knowledge k ON k.id = COALESCE(s.id, l.id)
        WHERE k.client_id = :c
        ORDER BY rank_score ASC, distance ASC
        LIMIT :n
    """)

    with engine.connect() as connection:
        return connection.execute(query, {
            "e": str(query_embedding),
            "c": client_id,
            "n": limit,
            "q": ts_query,
        }).fetchall()


def get_handoff_policy_candidates(
    client_id: int,
    labels: tuple[str, ...],
    window: int,
    limit: int,
) -> list[str]:
    """Entries whose first `window` characters contain one of `labels`."""
    conditions = " OR ".join(
        f"left(ltrim(content), :window) ILIKE :label{i}"
        for i in range(len(labels))
    )
    params = {f"label{i}": f"%{label}%" for i, label in enumerate(labels)}

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                SELECT content FROM knowledge
                WHERE client_id = :client_id AND ({conditions})
                ORDER BY id
                LIMIT :limit
            """),
            {"client_id": client_id, "window": window, "limit": limit, **params},
        ).fetchall()

    return [row.content for row in rows]
