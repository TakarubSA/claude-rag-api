import os
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

SEARCH_LIMIT = 20
CONTEXT_LIMIT = 8
MAX_DISTANCE = 0.65          # semantic-only rows must be this close
MAX_DISTANCE_KEYWORD = 0.80  # rows with an exact keyword hit may be farther
MAX_HISTORY_CHARS = 6000

USE_RERANK = False

NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
    "يشمل هذا السؤال الاستفسارات التالية",
)


# --------------------------------------------------
# Text helpers
# --------------------------------------------------


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_answer_format(answer: str) -> str:
    """Remove Markdown bold markers from customer-facing answers."""
    answer = (answer or "").strip()
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer, flags=re.DOTALL)
    answer = answer.replace("**", "")
    answer = re.sub(r"__(.*?)__", r"\1", answer, flags=re.DOTALL)
    answer = answer.replace("__", "")
    return answer.strip()


def _clean_content(content: str) -> str:
    content = content or ""
    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker, 1)[0]
    return content.strip()


# --------------------------------------------------
# Retrieval helpers
# --------------------------------------------------


def _search_many(texts: list[str], client_id: int):
    try:
        embeddings = create_embeddings(texts, input_type="query")

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} "
                f"for {len(texts)} queries"
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
        return None


def _merge(*result_lists):
    """Merge query results, deduplicate by id, keep the best rank."""
    best: dict[Any, Any] = {}

    def retrieval_order(result):
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


def _is_relevant(result) -> bool:
    distance = float(result.distance)
    hits = int(getattr(result, "keyword_hits", 0) or 0)

    if distance <= MAX_DISTANCE:
        return True

    return hits > 0 and distance <= MAX_DISTANCE_KEYWORD


def _dedupe_results(results):
    """Drop rows whose cleaned content is identical, before applying the limit."""
    out = []
    seen: set[str] = set()

    for result in results:
        content = _clean_content(result.content)
        if not content:
            continue

        key = re.sub(r"\s+", " ", content).strip().casefold()
        if key in seen:
            continue

        seen.add(key)
        out.append(result)

    return out


def _rerank(query: str, results):
    if not USE_RERANK or len(results) < 2:
        return results

    try:
        import voyageai

        api_key = os.getenv("VOYAGE_API_KEY")

        if not api_key:
            print("RERANK SKIPPED: VOYAGE_API_KEY is not configured")
            return results

        voyage_client = voyageai.Client(api_key=api_key)
        reranked = voyage_client.rerank(
            query=query,
            documents=[_clean_content(r.content) for r in results],
            model="rerank-2",
            top_k=min(CONTEXT_LIMIT, len(results)),
        )

        return [results[item.index] for item in reranked.results]

    except Exception as exc:
        print(f"RERANK ERROR: {exc}")
        return results


def _build_context(results) -> str:
    return "\n\n".join(
        _clean_content(r.content)
        for r in results
        if _clean_content(r.content)
    )


# --------------------------------------------------
# Grounded answer generation
# --------------------------------------------------


def _generate_grounded_answer(
    question: str,
    context: str,
    client_config: dict | None,
    best_distance=None,
    history: str = "",
):
    relevance = check_context_relevance(
        question=question,
        context=context,
        client_config=client_config,
        history=history,
    )

    print("CONTEXT RELEVANCE:", relevance)

    if relevance != "ANSWERABLE":
        return None

    answer = generate_answer(
        question=question,
        context=context,
        client_config=client_config,
        history=history,
    )

    if not isinstance(answer, str) or not answer.strip():
        print("GROUNDED ANSWER EMPTY: falling back to knowledge-gap handling")
        return None

    return {
        "answer": _clean_answer_format(answer),
        "best_distance": best_distance,
        "status": "KNOWN",
    }


# --------------------------------------------------
# Main RAG pipeline
# --------------------------------------------------


def answer_question(
    question: str,
    client_id: int,
    client_config: dict | None = None,
    conversation_context: str | None = None,
):
    print("CLIENT CONFIG LOADED:", bool(client_config))
    print(
        "CONVERSATION CONTEXT RECEIVED:",
        bool(conversation_context),
        "CHARS:",
        len(conversation_context or ""),
    )

    cleaned_question = _clean_text(question)

    if not cleaned_question:
        return {
            "answer": "فضلاً اكتب سؤالك حتى أتمكن من مساعدتك.",
            "best_distance": None,
            "status": "EMPTY_QUESTION",
        }

    history = (conversation_context or "")[-MAX_HISTORY_CHARS:]

    # 1) Rewrite into a retrieval query; resolves references from history
    #    and detects casual chat.
    search_query = rewrite_query(
        cleaned_question,
        client_config=client_config,
        history=history,
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
                "answer": _clean_answer_format(casual_answer),
                "best_distance": None,
                "status": "CASUAL",
            }

        search_query = cleaned_question

    # 2) Search with the rewritten and original wording.
    texts: list[str] = []
    for t in (search_query, cleaned_question):
        if t and t.casefold() not in {x.casefold() for x in texts}:
            texts.append(t)

    result_lists = _search_many(texts, client_id)
    search_failed = result_lists is None

    if search_failed:
        print("RETRIEVAL UNAVAILABLE: continuing with client configuration.")
        result_lists = []

    results = [r for r in _merge(*result_lists) if _is_relevant(r)]
    results = _dedupe_results(results)

    if results:
        results = _rerank(search_query, results)
        results = results[:CONTEXT_LIMIT]

        for result in results:
            print("ID:", result.id)
            print("DISTANCE:", result.distance)
            print("KEYWORD HITS:", getattr(result, "keyword_hits", 0))
            print("RANK SCORE:", getattr(result, "rank_score", result.distance))
            print("CONTENT:", result.content)
            print("---")
    else:
        print("NO RETRIEVED KNOWLEDGE")

    knowledge_context = _build_context(results)

    print("\n========== RETRIEVED CONTEXT ==========")
    print(knowledge_context if knowledge_context else "[empty]")
    print("========================================\n")

    best_distance = min(
        (float(r.distance) for r in results),
        default=None,
    )

    # 3) The model decides: clear question / subject from history /
    #    ask for the missing detail. Facts come only from the knowledge.
    grounded_result = _generate_grounded_answer(
        question=cleaned_question,
        context=knowledge_context,
        client_config=client_config,
        best_distance=best_distance,
        history=history,
    )

    if grounded_result:
        if search_failed or not results:
            grounded_result["status"] = "CONFIG_ONLY"
        return grounded_result

    # 4) Nothing usable: classify the gap and use the deterministic fallback.
    gap_type = classify_knowledge_gap(
        question=search_query,
        context=knowledge_context,
        client_config=client_config,
    )

    print("KNOWLEDGE GAP:", gap_type)

    fallback = generate_fallback_answer(
        question=cleaned_question,
        gap_type=gap_type,
        client_config=client_config,
    )

    if not isinstance(fallback, str) or not fallback.strip():
        fallback = (
            "عذرًا، لا تتوفر لدي معلومات مؤكدة للإجابة عن سؤالك حاليًا. "
            "يرجى التواصل مع خدمة العملاء للمساعدة."
        )

    return {
        "answer": _clean_answer_format(fallback),
        "best_distance": best_distance,
        "status": "SEARCH_ERROR" if search_failed else gap_type,
    }