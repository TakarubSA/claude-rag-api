import re

from app.embeddings.voyage import create_embeddings
from app.knowledge.repository import search_knowledge
from app.llm.claude import (
    rewrite_query,
    generate_answer,
    generate_casual_answer,
    generate_fallback_answer,
    check_context_relevance,
    classify_knowledge_gap,
)

CASUAL_MARKER = "__CASUAL_CONVERSATION__"

SEARCH_LIMIT = 30
CONTEXT_LIMIT = 6

USE_RERANK = False

NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
)


# ----------------------------------------------------------------------
# Text normalization
# ----------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = (text or "").strip()

    text = re.sub(
        r"[\u064B-\u0652\u0640]",
        "",
        text,
    )

    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text,
    )

    return re.sub(r"\s+", " ", text)


def _clean_content(content: str) -> str:
    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker)[0]

    return content.strip()


# ----------------------------------------------------------------------
# Generic tokenization
# ----------------------------------------------------------------------

STOPWORDS = {
    # Arabic
    "من",
    "ما",
    "ماذا",
    "متى",
    "اين",
    "وين",
    "كيف",
    "هل",
    "ايش",
    "وش",
    "شو",
    "كم",
    "عندكم",
    "عندي",
    "اللي",
    "الذي",
    "التي",
    "في",
    "فيه",
    "فيها",
    "على",
    "الى",
    "عن",
    "مع",
    "هذا",
    "هذه",
    "هو",
    "هي",
    "انا",
    "ابغى",
    "اريد",
    "ابي",
    "لو",
    "يا",
    "ياكم",

    # English
    "the",
    "a",
    "an",
    "is",
    "are",
    "what",
    "who",
    "where",
    "when",
    "how",
    "do",
    "does",
    "can",
    "you",
    "your",
    "for",
    "with",
    "from",
}


def _tokens(text: str) -> set[str]:
    text = _clean_text(text).lower()

    # Keep Arabic + Latin + numbers
    raw_tokens = re.findall(
        r"[\u0600-\u06FFa-zA-Z0-9_]+",
        text,
    )

    return {
        token
        for token in raw_tokens
        if token not in STOPWORDS
        and len(token) > 1
    }


# ----------------------------------------------------------------------
# Generic lexical relevance
# ----------------------------------------------------------------------

def _lexical_score(query: str, content: str) -> float:
    """
    Generic lexical relevance.

    No business-specific keywords.
    """

    query_tokens = _tokens(query)
    content_tokens = _tokens(content)

    if not query_tokens or not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens

    coverage = len(overlap) / len(query_tokens)

    # Reward exact normalized phrase presence.
    normalized_query = _clean_text(query).lower()
    normalized_content = _clean_text(content).lower()

    phrase_bonus = (
        0.35
        if normalized_query
        and normalized_query in normalized_content
        else 0.0
    )

    score = min(
        1.0,
        coverage + phrase_bonus,
    )

    return score


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------

def _search_many(texts, client_id: int):
    """
    Embed all queries in ONE Voyage request.

    This is important with low Voyage rate limits.
    """

    try:
        embeddings = create_embeddings(texts)

        return [
            search_knowledge(
                emb,
                client_id=client_id,
                limit=SEARCH_LIMIT,
            )
            for emb in embeddings
        ]

    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        return None


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

def _merge(*result_lists):
    """
    Merge results from multiple query variants.

    Keep the best vector distance for each knowledge ID.
    """

    best = {}

    for results in result_lists:
        for result in results:
            if (
                result.id not in best
                or result.distance < best[result.id].distance
            ):
                best[result.id] = result

    return list(best.values())


# ----------------------------------------------------------------------
# Generic hybrid ranking
# ----------------------------------------------------------------------

def _hybrid_rank(query: str, results):
    """
    Combine vector similarity with lexical similarity.

    This is domain-agnostic.
    """

    if not results:
        return []

    ranked = []

    for result in results:
        vector_score = max(
            0.0,
            1.0 - float(result.distance),
        )

        lexical_score = _lexical_score(
            query,
            _clean_content(result.content),
        )

        # Vector remains the main signal.
        # Lexical relevance fixes cases where semantic search returns
        # broadly related but not actually useful entries.
        final_score = (
            0.65 * vector_score
            + 0.35 * lexical_score
        )

        ranked.append(
            (
                final_score,
                vector_score,
                lexical_score,
                result,
            )
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        item[3]
        for item in ranked
    ]


# ----------------------------------------------------------------------
# Optional Voyage rerank
# ----------------------------------------------------------------------

def _rerank(query: str, results):
    """
    Optional external reranker.

    Disabled by default.
    """

    if not USE_RERANK or len(results) < 2:
        return results

    try:
        import os
        import voyageai

        vo = voyageai.Client(
            api_key=os.getenv("VOYAGE_API_KEY")
        )

        reranked = vo.rerank(
            query=query,
            documents=[
                _clean_content(r.content)
                for r in results
            ],
            model="rerank-2",
            top_k=min(
                CONTEXT_LIMIT,
                len(results),
            ),
        )

        return [
            results[item.index]
            for item in reranked.results
        ]

    except Exception as e:
        print(f"RERANK ERROR: {e}")
        return results


# ----------------------------------------------------------------------
# Main RAG flow
# ----------------------------------------------------------------------

def answer_question(
    question: str,
    client_id: int,
):
    cleaned_question = _clean_text(question)

    # 1. Understand the message.
    search_query = rewrite_query(question)

    print("ORIGINAL QUERY:", question)
    print("SEARCH QUERY:", search_query)

    # 2. Casual message -> no retrieval.
    if search_query == CASUAL_MARKER:
        return {
            "answer": generate_casual_answer(question),
            "best_distance": None,
            "status": "CASUAL",
        }

    # 3. Search using both rewritten query and original.
    texts = [search_query]

    if (
        cleaned_question
        and cleaned_question != search_query
    ):
        texts.append(cleaned_question)

    result_lists = _search_many(
        texts,
        client_id,
    )

    if result_lists is None:
        return {
            "answer": generate_fallback_answer(
                question,
                "RELATED_BUT_UNKNOWN",
            ),
            "best_distance": None,
            "status": "SEARCH_ERROR",
        }

    # 4. Merge all candidates.
    results = _merge(*result_lists)

    # 5. Generic hybrid ranking.
    results = _hybrid_rank(
        search_query,
        results,
    )

    # 6. Optional external reranker.
    results = _rerank(
        search_query,
        results,
    )

    # 7. Keep only the best context.
    results = results[:CONTEXT_LIMIT]

    for result in results:
        print("ID:", result.id)
        print("DISTANCE:", result.distance)
        print("CONTENT:", result.content)
        print("---")

    # 8. Build context.
    context = "\n\n".join(
        _clean_content(result.content)
        for result in results
    )

    print(
        "\n========== RETRIEVED CONTEXT =========="
    )
    print(context)
    print(
        "========================================\n"
    )

    best_distance = (
        results[0].distance
        if results
        else None
    )

    # 9. Nothing retrieved.
    if not results:
        gap_type = classify_knowledge_gap(
            question=question,
            context="",
        )

        print(
            "KNOWLEDGE GAP:",
            gap_type,
        )

        return {
            "answer": generate_fallback_answer(
                question,
                gap_type,
            ),
            "best_distance": None,
            "status": gap_type,
        }

    # 10. Check whether context can answer.
    relevance = check_context_relevance(
        question=question,
        context=context,
    )

    print(
        "CONTEXT RELEVANCE:",
        relevance,
    )

    if relevance == "ANSWERABLE":
        return {
            "answer": generate_answer(
                question=question,
                context=context,
            ),
            "best_distance": best_distance,
            "status": "KNOWN",
        }

    # 11. Knowledge gap.
    gap_type = classify_knowledge_gap(
        question=question,
        context=context,
    )

    print(
        "KNOWLEDGE GAP:",
        gap_type,
    )

    return {
        "answer": generate_fallback_answer(
            question,
            gap_type,
        ),
        "best_distance": best_distance,
        "status": gap_type,
    }