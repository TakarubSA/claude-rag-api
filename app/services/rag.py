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

NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
)


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text)


def _clean_content(content: str) -> str:
    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker)[0]

    return content.strip()


def _search_many(texts, client_id: int):
    """
    Embed all texts in ONE Voyage request,
    then search each query against this client's knowledge.
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


def _merge(*result_lists):
    """
    Merge results from several queries.
    Keep the best distance for each knowledge entry.
    """
    best = {}

    for results in result_lists:
        for r in results:
            if r.id not in best or r.distance < best[r.id].distance:
                best[r.id] = r

    return sorted(
        best.values(),
        key=lambda r: r.distance,
    )


def _rerank(query: str, results):
    """
    Optional Voyage reranking.
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


def answer_question(
    question: str,
    client_id: int,
    client_config: dict | None = None,
):
    print(
        "CLIENT CONFIG LOADED:",
        bool(client_config),
    )

    cleaned_question = _clean_text(question)

    # 1. Understand the message
    # client_config is now passed to the query understanding layer.
    search_query = rewrite_query(
        question,
        client_config=client_config,
    )

    print("ORIGINAL QUERY:", question)
    print("SEARCH QUERY:", search_query)

    # 2. Casual conversation
    if search_query == CASUAL_MARKER:
        return {
            "answer": generate_casual_answer(question),
            "best_distance": None,
            "status": "CASUAL",
        }

    # 3. Search with rewritten query + cleaned original
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

    # 4. Merge + optional rerank
    results = _merge(*result_lists)

    results = _rerank(
        search_query,
        results,
    )[:CONTEXT_LIMIT]

    for result in results:
        print("ID:", result.id)
        print("DISTANCE:", result.distance)
        print("CONTENT:", result.content)
        print("---")

    # 5. Build context
    context = "\n\n".join(
        _clean_content(r.content)
        for r in results
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

    # 6. No results
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

    # 7. Check whether context can answer
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

    # 8. Not answerable
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