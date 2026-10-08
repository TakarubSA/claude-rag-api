import re

from app.embeddings.voyage import create_embeddings  # batch version, see notes
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
SEARCH_LIMIT = 8        # results fetched per query
CONTEXT_LIMIT = 6       # entries sent to Claude
USE_RERANK = False      # set True after testing Voyage rerank (see _rerank)

# Knowledge entries sometimes contain a list of example questions after this
# marker. It pollutes the context, so we cut it before sending to Claude.
NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
)


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)   # diacritics / tatweel
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)           # هلاااا -> هلاا
    return re.sub(r"\s+", " ", text)


def _clean_content(content: str) -> str:
    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker)[0]
    return content.strip()


def _search_many(texts, client_id: int):
    """
    Embed all texts in ONE Voyage request (important with low rate limits),
    then search for each. Returns a list of result lists, or None if the
    embedding/search failed (so we never confuse an outage with 'no answer').
    """
    try:
        embeddings = create_embeddings(texts)
        return [
            search_knowledge(emb, client_id=client_id, limit=SEARCH_LIMIT)
            for emb in embeddings
        ]
    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        return None


def _merge(*result_lists):
    """Merge results from several queries. Keep the best distance per ID."""
    best = {}
    for results in result_lists:
        for r in results:
            if r.id not in best or r.distance < best[r.id].distance:
                best[r.id] = r
    return sorted(best.values(), key=lambda r: r.distance)


def _rerank(query: str, results):
    """
    Optional: rerank with Voyage. Falls back silently to distance order.
    Check the model name against your Voyage account/docs.
    """
    if not USE_RERANK or len(results) < 2:
        return results
    try:
        import os
        import voyageai

        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        reranked = vo.rerank(
            query=query,
            documents=[_clean_content(r.content) for r in results],
            model="rerank-2",
            top_k=min(CONTEXT_LIMIT, len(results)),
        )
        return [results[item.index] for item in reranked.results]
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

    # 1. Understand the message (typo fixing + casual detection)
    search_query = rewrite_query(question)

    print("ORIGINAL QUERY:", question)
    print("SEARCH QUERY:", search_query)
    cleaned_question = _clean_text(question)

    # 1. Understand the message (typo fixing + casual detection)
    search_query = rewrite_query(question)

    print("ORIGINAL QUERY:", question)
    print("SEARCH QUERY:", search_query)

    # 2. Casual chat (كيفك، ايش مسوي، شكرا...) -> no retrieval needed
    if search_query == CASUAL_MARKER:
        return {
            "answer": generate_casual_answer(question),
            "best_distance": None,
            "status": "CASUAL",
        }

    # 3. Search with BOTH the rewritten query and the cleaned original.
    #    If the rewrite is weak or wrong, the original still finds the entry.
    texts = [search_query]
    if cleaned_question and cleaned_question != search_query:
        texts.append(cleaned_question)

    result_lists = _search_many(texts, client_id)

    if result_lists is None:
        # Embedding/search service failed (e.g. rate limit). Do not guess and
        # do not report it as a knowledge gap: send the user to the website.
        return {
            "answer": generate_fallback_answer(question, "RELATED_BUT_UNKNOWN"),
            "best_distance": None,
            "status": "SEARCH_ERROR",
        }

    results = _merge(*result_lists)
    results = _rerank(search_query, results)[:CONTEXT_LIMIT]

    for result in results:
        print("ID:", result.id)
        print("DISTANCE:", result.distance)
        print("CONTENT:", result.content)
        print("---")

    # 4. Build context (noise removed)
    context = "\n\n".join(_clean_content(r.content) for r in results)

    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)
    print("========================================\n")

    best_distance = results[0].distance if results else None

    # 5. Nothing retrieved at all -> classify and fall back to the website
    if not results:
        gap_type = classify_knowledge_gap(question=question, context="")
        print("KNOWLEDGE GAP:", gap_type)
        return {
            "answer": generate_fallback_answer(question, gap_type),
            "best_distance": None,
            "status": gap_type,
        }

    # 6. Is the context able to answer the ORIGINAL question?
    relevance = check_context_relevance(question=question, context=context)
    print("CONTEXT RELEVANCE:", relevance)

    if relevance == "ANSWERABLE":
        return {
            "answer": generate_answer(question=question, context=context),
            "best_distance": best_distance,
            "status": "KNOWN",
        }

    # 7. Not answerable -> never guess. Point to the website / right service.
    gap_type = classify_knowledge_gap(question=question, context=context)
    print("KNOWLEDGE GAP:", gap_type)

    return {
        "answer": generate_fallback_answer(question, gap_type),
        "best_distance": best_distance,
        "status": gap_type,
    }