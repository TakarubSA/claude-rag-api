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

# Keep disabled until retrieval quality is tested properly.
USE_RERANK = False


# Knowledge entries can contain generated example questions.
# They add noise without necessarily adding evidence.
NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
)


def _clean_text(text: str) -> str:
    """
    Lightweight normalization for the user's original message.
    """

    text = (text or "").strip()

    # Remove Arabic diacritics and tatweel.
    text = re.sub(
        r"[\u064B-\u0652\u0640]",
        "",
        text,
    )

    # هلااااا -> هلاا
    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _clean_content(content: str) -> str:
    """
    Clean a retrieved knowledge entry before sending it to Claude.
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
    Run semantic retrieval for all queries in one embedding request.

    client_id is always passed to the repository so knowledge
    isolation remains tenant-specific.
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

        # IMPORTANT:
        # Retrieval failure does NOT mean the AI cannot answer.
        # The caller can still use client_config as a source of truth.
        return None


def _merge(*result_lists):
    """
    Merge retrieval results from multiple queries.

    If the same knowledge row appears multiple times,
    keep the best semantic distance.
    """

    best = {}

    for results in result_lists:
        if not results:
            continue

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
    Optional Voyage reranking.

    Disabled by default.
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


def _build_context(results) -> str:
    """
    Convert retrieved rows into clean LLM context.

    Returns an empty string when no useful knowledge was retrieved.
    """

    if not results:
        return ""

    cleaned_entries = []

    for result in results:
        content = _clean_content(
            result.content
        )

        if content:
            cleaned_entries.append(
                content
            )

    return "\n\n".join(
        cleaned_entries
    )


def _generate_grounded_answer(
    question: str,
    context: str,
    client_config: dict | None,
    best_distance=None,
):
    """
    Ask Claude to answer using BOTH:

    1. client_config
    2. retrieved knowledge

    The context is allowed to be empty.

    This is important because basic business questions such as
    website, company identity, workflows, and stable business
    rules can often be answered directly from client configuration.
    """

    relevance = check_context_relevance(
        question=question,
        context=context,
        client_config=client_config,
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
                client_config=client_config,
            ),
            "best_distance": best_distance,
            "status": "KNOWN",
        }

    return None


def answer_question(
    question: str,
    client_id: int,
    client_config: dict | None = None,
):
    """
    Main multi-tenant RAG orchestration.

    Important architectural rule:

        Retrieved knowledge is evidence.
        Client configuration is also evidence.

    Retrieval failure or an empty knowledge result must NOT
    automatically mean that the assistant cannot answer.

    Pipeline:

        user question
            ↓
        query understanding
            ↓
        semantic retrieval
            ↓
        context construction
            ↓
        client_config + context
            ↓
        evidence evaluation
            ↓
        grounded answer
            OR
        knowledge-gap classification
            ↓
        deterministic fallback
    """

    print(
        "CLIENT CONFIG LOADED:",
        bool(client_config),
    )

    # --------------------------------------------------------------
    # 1. Normalize original question
    # --------------------------------------------------------------

    cleaned_question = _clean_text(
        question
    )

    # --------------------------------------------------------------
    # 2. Understand the query
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

    # Also keep the cleaned original.
    # This protects against a rewrite that accidentally loses
    # an important keyword/entity.
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

    search_failed = (
        result_lists is None
    )

    if search_failed:
        print(
            "RETRIEVAL UNAVAILABLE:"
            " continuing with client configuration."
        )

        result_lists = []

    # --------------------------------------------------------------
    # 6. Merge results
    # --------------------------------------------------------------

    results = _merge(
        *result_lists
    )

    # --------------------------------------------------------------
    # 7. Optional reranking
    # --------------------------------------------------------------

    if results:
        results = _rerank(
            search_query,
            results,
        )

        results = results[
            :CONTEXT_LIMIT
        ]

    # --------------------------------------------------------------
    # 8. Debug retrieved evidence
    # --------------------------------------------------------------

    if results:
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
    else:
        print(
            "NO RETRIEVED KNOWLEDGE"
        )

    # --------------------------------------------------------------
    # 9. Build context
    # --------------------------------------------------------------

    context = _build_context(
        results
    )

    print(
        "\n========== RETRIEVED CONTEXT =========="
    )

    print(
        context
        if context
        else "[empty]"
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
    # 10. Try answering from ALL available evidence
    #
    #     This is the important part:
    #
    #     context can be empty.
    #     Claude still receives client_config.
    # --------------------------------------------------------------

    grounded_result = _generate_grounded_answer(
        question=question,
        context=context,
        client_config=client_config,
        best_distance=best_distance,
    )

    if grounded_result:
        # If retrieval failed but configuration was enough,
        # this is still a valid known answer.
        if search_failed:
            grounded_result["status"] = "CONFIG_ONLY"

        elif not results:
            grounded_result["status"] = "CONFIG_ONLY"

        return grounded_result

    # --------------------------------------------------------------
    # 11. We don't have enough evidence
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
    # 12. Safe deterministic fallback
    # --------------------------------------------------------------

    return {
        "answer": generate_fallback_answer(
            question=question,
            gap_type=gap_type,
            client_config=client_config,
        ),
        "best_distance": best_distance,
        "status": (
            "SEARCH_ERROR"
            if search_failed
            else gap_type
        ),
    }