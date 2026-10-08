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

SEARCH_LIMIT = 8
CONTEXT_LIMIT = 6
USE_RERANK = False


# Knowledge entries sometimes contain example questions after this marker.
# They can pollute the context, so we remove them before sending context
# to the LLM.
NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
)


def _clean_text(text: str) -> str:
    """
    Lightweight normalization of the user's original message.
    """

    text = (text or "").strip()

    # Remove Arabic diacritics and tatweel.
    text = re.sub(
        r"[\u064B-\u0652\u0640]",
        "",
        text,
    )

    # Collapse exaggerated repeated characters:
    # هلااااا -> هلاا
    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text,
    )

    # Collapse whitespace.
    return re.sub(
        r"\s+",
        " ",
        text,
    )


def _clean_content(content: str) -> str:
    """
    Clean retrieved knowledge before sending it to Claude.
    """

    content = content or ""

    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker)[0]

    return content.strip()


def _search_many(
    texts,
    client_id: int,
):
    """
    Embed all retrieval queries in a single Voyage request,
    then search each query against the current client's knowledge.

    Multi-tenant isolation remains enforced by client_id.
    """

    try:
        embeddings = create_embeddings(texts)

        return [
            search_knowledge(
                embedding,
                client_id=client_id,
                limit=SEARCH_LIMIT,
            )
            for embedding in embeddings
        ]

    except Exception as e:
        print(
            f"SEARCH ERROR: {e}"
        )
        return None


def _merge(*result_lists):
    """
    Merge results from multiple retrieval queries.

    If the same knowledge row appears more than once,
    keep the best semantic distance.
    """

    best = {}

    for results in result_lists:
        for result in results:
            if (
                result.id not in best
                or result.distance < best[result.id].distance
            ):
                best[result.id] = result

    return sorted(
        best.values(),
        key=lambda result: result.distance,
    )


def _rerank(
    query: str,
    results,
):
    """
    Optional Voyage reranking layer.

    Disabled by default until explicitly enabled and tested.
    """

    if not USE_RERANK or len(results) < 2:
        return results

    try:
        import os
        import voyageai

        voyage_client = voyageai.Client(
            api_key=os.getenv("VOYAGE_API_KEY")
        )

        reranked = voyage_client.rerank(
            query=query,
            documents=[
                _clean_content(result.content)
                for result in results
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
        print(
            f"RERANK ERROR: {e}"
        )

        return results


def answer_question(
    question: str,
    client_id: int,
    client_config: dict | None = None,
):
    """
    Main RAG orchestration pipeline.

    Flow:

        user question
            ↓
        query understanding
            ↓
        vector retrieval
            ↓
        merge
            ↓
        optional rerank
            ↓
        context construction
            ↓
        relevance evaluation
            ↓
        grounded answer
            OR
        knowledge-gap classification
            ↓
        safe fallback

    client_config is passed to every LLM layer so that the
    pipeline remains completely tenant-aware without hardcoding
    any specific business.
    """

    print(
        "CLIENT CONFIG LOADED:",
        bool(client_config),
    )

    # --------------------------------------------------------------
    # 1. Clean original question
    # --------------------------------------------------------------

    cleaned_question = _clean_text(
        question
    )

    # --------------------------------------------------------------
    # 2. Understand / rewrite the question
    # --------------------------------------------------------------

    search_query = rewrite_query(
        question,
        client_config=client_config,
    )

    print(
        "ORIGINAL QUERY:",
        question,
    )

    print(
        "SEARCH QUERY:",
        search_query,
    )

    # --------------------------------------------------------------
    # 3. Casual conversation
    # --------------------------------------------------------------

    if search_query == CASUAL_MARKER:
        return {
            "answer": generate_casual_answer(
                question,
                client_config=client_config,
            ),
            "best_distance": None,
            "status": "CASUAL",
        }

    # --------------------------------------------------------------
    # 4. Build retrieval queries
    # --------------------------------------------------------------

    texts = [
        search_query,
    ]

    # Keep the original normalized message as a second retrieval query.
    # This protects against a weak or overly aggressive rewrite.
    if (
        cleaned_question
        and cleaned_question != search_query
    ):
        texts.append(
            cleaned_question
        )

    # --------------------------------------------------------------
    # 5. Retrieve knowledge
    # --------------------------------------------------------------

    result_lists = _search_many(
        texts,
        client_id,
    )

    # Search / embedding failure is not the same thing as
    # "knowledge does not exist".
    if result_lists is None:
        return {
            "answer": generate_fallback_answer(
                question=question,
                gap_type="RELATED_BUT_UNKNOWN",
                client_config=client_config,
            ),
            "best_distance": None,
            "status": "SEARCH_ERROR",
        }

    # --------------------------------------------------------------
    # 6. Merge retrieval results
    # --------------------------------------------------------------

    results = _merge(
        *result_lists
    )

    # --------------------------------------------------------------
    # 7. Optional reranking
    # --------------------------------------------------------------

    results = _rerank(
        search_query,
        results,
    )

    results = results[:CONTEXT_LIMIT]

    # --------------------------------------------------------------
    # 8. Debug retrieved knowledge
    # --------------------------------------------------------------

    for result in results:
        print(
            "ID:",
            result.id,
        )

        print(
            "DISTANCE:",
            result.distance,
        )

        print(
            "CONTENT:",
            result.content,
        )

        print("---")

    # --------------------------------------------------------------
    # 9. Build clean context
    # --------------------------------------------------------------

    context = "\n\n".join(
        _clean_content(result.content)
        for result in results
    )

    print(
        "\n========== RETRIEVED CONTEXT =========="
    )

    print(
        context
    )

    print(
        "========================================\n"
    )

    best_distance = (
        results[0].distance
        if results
        else None
    )

    # --------------------------------------------------------------
    # 10. Nothing retrieved
    # --------------------------------------------------------------

    if not results:
        gap_type = classify_knowledge_gap(
            question=question,
            context="",
            client_config=client_config,
        )

        print(
            "KNOWLEDGE GAP:",
            gap_type,
        )

        return {
            "answer": generate_fallback_answer(
                question=question,
                gap_type=gap_type,
                client_config=client_config,
            ),
            "best_distance": None,
            "status": gap_type,
        }

    # --------------------------------------------------------------
    # 11. Evaluate context relevance
    # --------------------------------------------------------------

    relevance = check_context_relevance(
        question=question,
        context=context,
        client_config=client_config,
    )

    print(
        "CONTEXT RELEVANCE:",
        relevance,
    )

    # --------------------------------------------------------------
    # 12. Generate grounded answer
    # --------------------------------------------------------------

    if relevance == "ANSWERABLE":
        return {
            "answer": generate_answer(
                question=question,
                context=context,
                client_config=client_config,
            ),
            "best_distance": best_distance,
            "status": "KNOWN",
        }

    # --------------------------------------------------------------
    # 13. Context not sufficient
    # --------------------------------------------------------------

    gap_type = classify_knowledge_gap(
        question=question,
        context=context,
        client_config=client_config,
    )

    print(
        "KNOWLEDGE GAP:",
        gap_type,
    )

    # --------------------------------------------------------------
    # 14. Safe fallback
    # --------------------------------------------------------------

    return {
        "answer": generate_fallback_answer(
            question=question,
            gap_type=gap_type,
            client_config=client_config,
        ),
        "best_distance": best_distance,
        "status": gap_type,
    }