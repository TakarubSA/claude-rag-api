
import re

from sqlalchemy import text

from app.database.connection import engine


# Normalize common Arabic spelling variants for keyword retrieval.
_ARABIC_TRANSLATION = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
})


_STOP_WORDS = {
    "هل", "من", "في", "على", "عن", "مع", "او", "ما", "ماذا",
    "كيف", "متى", "وين", "اين", "ايش", "وش", "مين", "كم",
    "ليش", "لماذا", "انا", "انت", "عندي", "عندنا", "عندكم",
    "لدينا", "لديكم", "ممكن", "لو", "ابغى", "ابي", "اريد",
    "احتاج", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي",
    "هو", "هي", "هم", "نحن", "كان", "كانت", "يكون", "تكون",
    "اللي", "بس", "يعني", "عشان", "فقط", "ثم", "و",
    "the", "and", "for", "are", "you", "your", "what", "how",
    "can", "with", "from",
    "حكيم", "كير", "طبيب", "طبي", "طبية", "طبيه",
    "اطباء", "دواء", "ادوية", "خدمة", "استشارة", "استشاره",
    "دكتور", "دكتورة", "الدكتورة",
}


def _normalize_arabic(value: str) -> str:
    """Normalize common Arabic letter variants and diacritics."""
    value = re.sub(
        r"[\u064B-\u065F\u0670\u0640]",
        "",
        value.lower(),
    )
    return value.translate(_ARABIC_TRANSLATION)


def _extract_keywords(query_text: str | None) -> list[str]:
    """Extract distinctive terms for lexical retrieval."""
    if not query_text:
        return []

    normalized = _normalize_arabic(query_text)
    tokens = re.findall(
        r"[\u0600-\u06FF]+|[a-z0-9]+",
        normalized,
    )

    keywords = []
    seen = set()

    for token in tokens:
        # Strip the Arabic definite article for comparison.
        if token.startswith("ال") and len(token) > 4:
            token = token[2:]

        if (
            len(token) < 3
            or token in _STOP_WORDS
            or token in seen
        ):
            continue

        seen.add(token)
        keywords.append(token)

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
    """
    Hybrid retrieval combining semantic similarity and whole-word matches.

    Every search path is restricted to the requested client_id.
    Results expose distance, keyword_hits and rank_score.
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
            ORDER BY
                embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        ),
        lexical AS (
            SELECT
                k.id,
                k.content,
                k.embedding <=> CAST(:embedding AS vector) AS distance,
                (
                    SELECT COUNT(*)
                    FROM unnest(CAST(:keywords AS TEXT[]))
                        AS kw(keyword)
                    WHERE EXISTS (
                        SELECT 1
                        FROM regexp_split_to_table(
                            translate(
                                lower(k.content),
                                'أإآٱى',
                                'ااااي'
                            ),
                            '[^a-z0-9ء-ي]+'
                        ) AS words(token)
                        WHERE
                            words.token = kw.keyword
                            OR (
                                words.token LIKE 'ال%'
                                AND substring(words.token FROM 3)
                                    = kw.keyword
                            )
                    )
                ) AS keyword_hits
            FROM knowledge AS k
            WHERE
                k.client_id = :client_id
                AND cardinality(CAST(:keywords AS TEXT[])) > 0
                AND EXISTS (
                    SELECT 1
                    FROM unnest(CAST(:keywords AS TEXT[]))
                        AS kw(keyword)
                    WHERE EXISTS (
                        SELECT 1
                        FROM regexp_split_to_table(
                            translate(
                                lower(k.content),
                                'أإآٱى',
                                'ااااي'
                            ),
                            '[^a-z0-9ء-ي]+'
                        ) AS words(token)
                        WHERE
                            words.token = kw.keyword
                            OR (
                                words.token LIKE 'ال%'
                                AND substring(words.token FROM 3)
                                    = kw.keyword
                            )
                    )
                )
            ORDER BY
                keyword_hits DESC,
                k.embedding <=> CAST(:embedding AS vector)
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
            distance - LEAST(
                0.12,
                keyword_hits * 0.04
            ) AS rank_score
        FROM deduplicated
        ORDER BY
            rank_score ASC,
            keyword_hits DESC,
            distance ASC
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
