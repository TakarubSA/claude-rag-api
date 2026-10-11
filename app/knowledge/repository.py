
import re

from sqlalchemy import text

from app.database.connection import engine


# Generic Arabic and English stop words. These are used only for
# lexical retrieval, not for changing the user's original question.
_STOP_WORDS = {
    "هل", "من", "في", "على", "عن", "الى", "إلى", "مع", "او", "أو",
    "ما", "ماذا", "كيف", "متى", "وين", "أين", "ايش", "إيش", "وش",
    "مين", "كم", "ليش", "لماذا", "انا", "أنا", "انت", "أنت", "عندي",
    "عندنا", "عندكم", "لدينا", "لديكم", "ممكن", "لو", "ابغى", "أبغى",
    "ابي", "أبي", "اريد", "أريد", "احتاج", "أحتاج", "هذا", "هذه",
    "ذلك", "تلك", "الذي", "التي", "هو", "هي", "هم", "نحن", "كان",
    "كانت", "يكون", "تكون", "اللي", "بس", "يعني", "عشان", "فقط",
    "ثم", "و", "the", "and", "for", "are", "you", "your", "what",
    "how", "can", "with", "from",
    # Common domain words are weak signals on their own.
    "حكيم", "كير", "طبيب", "الطبيب", "اطباء", "أطباء",
    "الاطباء", "الأطباء", "دواء", "الدواء", "الأدوية", "الادوية",
    "خدمة", "الخدمة", "استشارة", "استشاره", "طبية", "الطبية",
}


def _extract_keywords(query_text: str | None) -> list[str]:
    """Extract distinctive Arabic/English terms for lexical retrieval."""
    if not query_text:
        return []

    query_text = re.sub(
        r"[\u064B-\u065F\u0670\u0640]",
        "",
        query_text.lower(),
    )

    tokens = re.findall(r"[\u0600-\u06FF]+|[a-z0-9]+", query_text)

    keywords = []
    seen = set()

    for token in tokens:
        if len(token) < 3 or token in _STOP_WORDS or token in seen:
            continue

        seen.add(token)
        keywords.append(token)

    # Keep SQL work bounded for long messages.
    return keywords[:12]


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
    """
    Hybrid retrieval:
    1. Retrieve the nearest vector matches.
    2. Retrieve exact keyword matches.
    3. Merge candidates and boost distinctive lexical matches.

    Tenant isolation is enforced in BOTH retrieval paths.
    `distance` remains the real vector distance.
    `rank_score` is only used to order combined candidates.
    """
    keywords = _extract_keywords(query_text)

    query = text("""
        WITH semantic AS (
            SELECT
                id,
                content,
                embedding <=> CAST(:embedding AS vector) AS distance,
                0::INTEGER AS keyword_hits
            FROM knowledge
            WHERE client_id = :client_id
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        ),
        lexical AS (
            SELECT
                k.id,
                k.content,
                k.embedding <=> CAST(:embedding AS vector) AS distance,
                (
                    SELECT COUNT(*)
                    FROM unnest(CAST(:keywords AS TEXT[])) AS kw(keyword)
                    WHERE k.content ILIKE '%' || kw.keyword || '%'
                ) AS keyword_hits
            FROM knowledge AS k
            WHERE k.client_id = :client_id
              AND cardinality(CAST(:keywords AS TEXT[])) > 0
              AND EXISTS (
                  SELECT 1
                  FROM unnest(CAST(:keywords AS TEXT[])) AS kw(keyword)
                  WHERE k.content ILIKE '%' || kw.keyword || '%'
              )
            ORDER BY
                keyword_hits DESC,
                k.embedding <=> CAST(:embedding AS vector) ASC
            LIMIT :limit
        ),
        candidates AS (
            SELECT id, content, distance, keyword_hits
            FROM semantic

            UNION ALL

            SELECT id, content, distance, keyword_hits
            FROM lexical
        ),
        deduplicated AS (
            SELECT DISTINCT ON (id)
                id,
                content,
                distance,
                keyword_hits
            FROM candidates
            ORDER BY
                id,
                keyword_hits DESC,
                distance ASC
        )
        SELECT
            id,
            content,
            distance,
            keyword_hits,
            distance - CASE
                WHEN keyword_hits > 0 THEN
                    LEAST(
                        0.20,
                        0.16 + ((keyword_hits - 1) * 0.02)
                    )
                ELSE 0
            END AS rank_score
        FROM deduplicated
        ORDER BY rank_score ASC, distance ASC
        LIMIT :limit
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "embedding": str(query_embedding),
                "client_id": client_id,
                "limit": limit,
                "keywords": keywords,
            },
        )

        return result.fetchall()
