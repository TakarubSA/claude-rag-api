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
CONTEXT_LIMIT = 6
MAX_DISTANCE = 0.65
MAX_HISTORY_CHARS = 6000

USE_RERANK = False

NOISE_MARKERS = (
    "هذه المعلومة تشمل أسئلة مثل",
)


# --------------------------------------------------
# Text helpers
# --------------------------------------------------


def _clean_text(text: str) -> str:
    """Normalize text without changing its intended meaning."""
    text = (text or "").strip()

    # Remove Arabic diacritics and tatweel.
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)

    # Reduce elongated characters.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse whitespace and newlines.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _clean_answer_format(answer: str) -> str:
    """Clean formatting and hide internal RAG terminology from customer-facing answers."""
    answer = (answer or "").strip()

    # Remove paired bold markers while preserving their inner text.
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer, flags=re.DOTALL)
    answer = answer.replace("**", "")

    # Also remove Markdown bold markers written with underscores.
    answer = re.sub(r"__(.*?)__", r"\1", answer, flags=re.DOTALL)
    answer = answer.replace("__", "")

    # Replace internal technical terms if the model accidentally exposes them.
    internal_terms = (
        ("المعلومات المسترجعة", "المعلومات المتاحة"),
        ("البيانات المسترجعة", "المعلومات المتاحة"),
        ("السياق", "المعلومات المتاحة"),
    )
    for internal_term, customer_friendly_term in internal_terms:
        answer = answer.replace(internal_term, customer_friendly_term)

    return answer.strip()


def _clean_content(content: str) -> str:
    """Remove known non-evidentiary suffixes from retrieved knowledge."""
    content = content or ""

    for marker in NOISE_MARKERS:
        if marker in content:
            content = content.split(marker, 1)[0]

    return content.strip()


def _build_contextual_question(
    question: str,
    conversation_context: str | None,
) -> str:
    """Explain how the latest message should be interpreted using history."""
    history = (conversation_context or "").strip()

    if not history:
        return question

    history = history[-MAX_HISTORY_CHARS:]

    return f"""
Use the recent conversation to understand the user's latest message.

The conversation history is provided to resolve references, omitted
subjects, follow-up questions, and the meaning of short replies.

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE:
{question}

INSTRUCTIONS:
- Answer the latest user message only.
- Use the conversation history to resolve what words such as
  "he", "it", "that service", "how much", and "is it a specialist"
  refer to.
- Do not repeat the previous answer unless it is necessary.
- Treat prior assistant messages as conversation history, not as
  independently verified business facts.
- Use the retrieved knowledge and client configuration to verify
  business facts.
- Do not invent prices, qualifications, policies, or other details.
- Never mention internal technical terms such as "السياق" or "المعلومات المسترجعة".
- Speak naturally as a customer-service agent. If information is unavailable, say so simply without explaining internal system mechanics.
- If the available knowledge does not answer the latest question,
  say so clearly and direct the user to the appropriate support channel.
- Write in the user's language.
- Return plain text. Do not use Markdown bold markers or double asterisks.
""".strip()


# --------------------------------------------------
# Retrieval helpers
# --------------------------------------------------


def _search_many(texts: list[str], client_id: int):
    """Search the current tenant's knowledge for each query."""
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
    """Merge query results, deduplicate records, and keep the best rank."""
    best: dict[Any, Any] = {}

    def retrieval_order(result):
        score = getattr(result, "rank_score", result.distance)
        keyword_hits = int(getattr(result, "keyword_hits", 0) or 0)
        distance = float(result.distance)

        return (
            float(score),
            -keyword_hits,
            distance,
        )

    for results in result_lists:
        if not results:
            continue

        for result in results:
            current = best.get(result.id)

            if current is None or retrieval_order(result) < retrieval_order(current):
                best[result.id] = result

    return sorted(best.values(), key=retrieval_order)


def _rerank(query: str, results):
    """Optionally rerank candidates with Voyage."""
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
            documents=[_clean_content(result.content) for result in results],
            model="rerank-2",
            top_k=min(CONTEXT_LIMIT, len(results)),
        )

        return [results[item.index] for item in reranked.results]

    except Exception as exc:
        print(f"RERANK ERROR: {exc}")
        return results


def _build_context(results) -> str:
    """Build clean, deduplicated context from retrieved knowledge."""
    if not results:
        return ""

    cleaned_entries: list[str] = []
    seen: set[str] = set()

    for result in results:
        content = _clean_content(result.content)

        if not content:
            continue

        dedupe_key = re.sub(r"\s+", " ", content).strip().casefold()

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        cleaned_entries.append(content)

    return "\n\n".join(cleaned_entries)


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
    """Let Claude answer from retrieved knowledge instead of pre-rejecting it."""
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
    """
    Multi-tenant RAG pipeline with conversation-aware retrieval.

    The latest user message is interpreted in context. Knowledge retrieval
    remains isolated by client_id.
    """
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

    # Keep the latest message and conversation history separate.
    history = (conversation_context or "").strip()[-MAX_HISTORY_CHARS:]

    # Claude resolves references using history and creates a standalone search query.
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

    # Handle ordinary casual conversation.
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

    # Search with the rewritten query and, when different, the original wording.
    # The original wording can preserve acronyms, names, and terms that the
    # rewriter may accidentally replace (for example, SIBO / "سيبو").
    texts = [search_query]

    if (
        cleaned_question
        and cleaned_question.casefold() != search_query.casefold()
    ):
        texts.append(cleaned_question)

    result_lists = _search_many(texts, client_id)
    search_failed = result_lists is None

    if search_failed:
        print(
            "RETRIEVAL UNAVAILABLE: "
            "continuing with client configuration."
        )
        result_lists = []

    results = [
        result
        for result in _merge(*result_lists)
        if (
            float(result.distance) <= MAX_DISTANCE
            or int(getattr(result, "keyword_hits", 0) or 0) > 0
        )
    ]

    if results:
        results = _rerank(search_query, results)
        results = results[:CONTEXT_LIMIT]

    if results:
        for result in results:
            print("ID:", result.id)
            print("DISTANCE:", result.distance)
            print("KEYWORD HITS:", getattr(result, "keyword_hits", 0))
            print(
                "RANK SCORE:",
                getattr(result, "rank_score", result.distance),
            )
            print("CONTENT:", result.content)
            print("---")
    else:
        print("NO RETRIEVED KNOWLEDGE")

    knowledge_context = _build_context(results)

    print("\n========== RETRIEVED CONTEXT ==========")
    print(knowledge_context if knowledge_context else "[empty]")
    print("========================================\n")

    best_distance = min(
        (float(result.distance) for result in results),
        default=None,
    )

    # Claude gets the actual latest message and history as separate inputs.
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

        grounded_result["answer"] = _clean_answer_format(
            grounded_result["answer"]
        )
        return grounded_result

    gap_type = classify_knowledge_gap(
        question=cleaned_question,
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
