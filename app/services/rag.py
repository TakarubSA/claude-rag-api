import re
from typing import Any

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
    """Normalize the user's original message without changing its meaning."""
    text = (text or "").strip()

    # Remove Arabic diacritics and tatweel.
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)

    # Reduce elongated characters: هلااااا -> هلاا
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_content(content: str) -> str:
    """Remove known non-evidentiary suffixes from retrieved knowledge."""
    content = content or ""

    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker, 1)[0]

    return content.strip()


def _search_many(texts: list[str], client_id: int):
    """Run hybrid retrieval for every query using its text and embedding."""
    try:
        embeddings = create_embeddings(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} for {len(texts)} queries"
            )

        return [
            search_knowledge(
                embedding,
                client_id=client_id,
                limit=SEARCH_LIMIT,
                query_text=query_text,
            )
            for query_text, embedding in zip(texts, embeddings)
        ]
    except Exception as exc:
        print(f"SEARCH ERROR: {exc}")
        # Retrieval failure does not automatically mean the AI cannot answer;
        # answer_question can still evaluate client_config.
        return None


def _merge(*result_lists):
    """Merge query results, deduplicate rows, and keep the best retrieval rank."""
    best: dict[Any, Any] = {}

    def retrieval_order(result):
        # rank_score already combines semantic distance with bounded lexical
        # evidence. It must be primary; keyword count is only a tie-breaker.
        score = getattr(result, "rank_score", result.distance)
        keyword_hits = int(getattr(result, "keyword_hits", 0) or 0)
        distance = float(result.distance)
        return (float(score), -keyword_hits, distance)

    for results in result_lists:
        if not results:
            continue

        for result in results:
            current = best.get(result.id)
            if current is None or retrieval_order(result) < retrieval_order(current):
                best[result.id] = result

    return sorted(best.values(), key=retrieval_order)


def _rerank(query: str, results):
    """Optionally rerank candidates with Voyage; disabled by default."""
    if not USE_RERANK or len(results) < 2:
        return results

    try:
        import os
        import voyageai

        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            print("RERANK SKIPPED: VOYAGE_API_KEY is not configured")
            return results

        voyage_client = voyageai.Client(api_key=api_key)
        reranked = voyage_client.rerank(
            query=query,
            documents=[_clean_content(result.content) for result in results],
            model="rerank-2",
            top_k=min(CONTEXT_LIMIT, len(results)),
        )

        return [results[item.index] for item in reranked.results]
    except Exception as exc:
        print(f"RERANK ERROR: {exc}")
        return results


def _build_context(results) -> str:
    """Convert retrieved rows into clean, deduplicated LLM context."""
    if not results:
        return ""

    cleaned_entries: list[str] = []
    seen: set[str] = set()

    for result in results:
        content = _clean_content(result.content)
        if not content:
            continue

        # Deduplicate identical content without changing the source records.
        dedupe_key = re.sub(r"\s+", " ", content).strip().casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        cleaned_entries.append(content)

    return "\n\n".join(cleaned_entries)


def _generate_grounded_answer(
    question: str,
    context: str,
    client_config: dict | None,
    best_distance=None,
):
    """Answer only when Claude finds sufficient evidence in context/config."""
    relevance = check_context_relevance(
        question=question,
        context=context,
        client_config=client_config,
    )
    print("CONTEXT RELEVANCE:", relevance)

    if relevance != "ANSWERABLE":
        return None

    answer = generate_answer(
        question=question,
        context=context,
        client_config=client_config,
    )

    if not isinstance(answer, str) or not answer.strip():
        print("GROUNDED ANSWER EMPTY: falling back to knowledge-gap handling")
        return None

    return {
        "answer": answer.strip(),
        "best_distance": best_distance,
        "status": "KNOWN",
    }


def answer_question(
    question: str,
    client_id: int,
    client_config: dict | None = None,
):
    """Multi-tenant RAG orchestration using retrieved knowledge and client config."""
    print("CLIENT CONFIG LOADED:", bool(client_config))

    cleaned_question = _clean_text(question)
    if not cleaned_question:
        return {
            "answer": "فضلاً اكتب سؤالك حتى أتمكن من مساعدتك.",
            "best_distance": None,
            "status": "EMPTY_QUESTION",
        }

    # Query rewriting helps retrieval, while the cleaned original is retained
    # as a second query so important terms are not lost during rewriting.
    search_query = rewrite_query(
        cleaned_question,
        client_config=client_config,
    )
    if not isinstance(search_query, str) or not search_query.strip():
        search_query = cleaned_question
    else:
        search_query = search_query.strip()

    print("ORIGINAL QUERY:", cleaned_question)
    print("SEARCH QUERY:", search_query)

    if search_query == CASUAL_MARKER:
        casual_answer = generate_casual_answer(
            cleaned_question,
            client_config=client_config,
        )
        if isinstance(casual_answer, str) and casual_answer.strip():
            return {
                "answer": casual_answer.strip(),
                "best_distance": None,
                "status": "CASUAL",
            }
        # Do not return an empty answer if the casual-answer call failed.
        search_query = cleaned_question

    texts = [search_query]
    if cleaned_question and cleaned_question != search_query:
        texts.append(cleaned_question)

    result_lists = _search_many(texts, client_id)
    search_failed = result_lists is None

    if search_failed:
        print("RETRIEVAL UNAVAILABLE: continuing with client configuration.")
        result_lists = []

    results = _merge(*result_lists)

    if results:
        results = _rerank(search_query, results)
        results = results[:CONTEXT_LIMIT]

    # Useful diagnostics for tuning retrieval in production.
    if results:
        for result in results:
            print("ID:", result.id)
            print("DISTANCE:", result.distance)
            print("KEYWORD HITS:", getattr(result, "keyword_hits", 0))
            print("RANK SCORE:", getattr(result, "rank_score", result.distance))
            print("CONTENT:", result.content)
            print("---")
    else:
        print("NO RETRIEVED KNOWLEDGE")

    context = _build_context(results)
    print("\n========== RETRIEVED CONTEXT ==========")
    print(context if context else "[empty]")
    print("========================================\n")

    # Preserve a true best semantic distance even if hybrid ranking reorders rows.
    best_distance = min(
        (float(result.distance) for result in results),
        default=None,
    )

    grounded_result = _generate_grounded_answer(
        question=cleaned_question,
        context=context,
        client_config=client_config,
        best_distance=best_distance,
    )

    if grounded_result:
        if search_failed or not results:
            grounded_result["status"] = "CONFIG_ONLY"
        return grounded_result

    gap_type = classify_knowledge_gap(
        question=cleaned_question,
        context=context,
        client_config=client_config,
    )
    print("KNOWLEDGE GAP:", gap_type)

    fallback = generate_fallback_answer(
        question=cleaned_question,
        gap_type=gap_type,
        client_config=client_config,
    )

    if not isinstance(fallback, str) or not fallback.strip():
        fallback = "عذرًا، لا تتوفر لدي معلومات مؤكدة للإجابة عن سؤالك حاليًا. يرجى التواصل مع خدمة العملاء للمساعدة."

    return {
        "answer": fallback.strip(),
        "best_distance": best_distance,
        "status": "SEARCH_ERROR" if search_failed else gap_type,
    }
