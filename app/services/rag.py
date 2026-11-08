import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.embeddings.voyage import create_embeddings
from app.knowledge.repository import (
    get_handoff_policy_candidates,
    search_knowledge,
)
from app.services.history import trim_history
from app.llm.claude import (
    CASUAL_MARKER,
    rewrite_query,
    generate_answer,
    generate_casual_answer,
    generate_fallback_answer,
    evaluate_question,
    classify_knowledge_gap,
)


logger = logging.getLogger(__name__)

# Two pools so a task never waits on work queued behind it in its own pool
# (which deadlocks under load):
# - executor: request-level Claude calls run alongside the request thread.
# - search_executor: DB searches only; never waits on anything itself.
#   Kept within the SQLAlchemy connection pool (5 + 10 overflow).
executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="rag")
search_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="search")

SEARCH_LIMIT = 60  # many entries are rephrasings of one answer; see _unique_answers
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


def _is_short_confirmation(message: str) -> bool:
    """Recognize standalone acknowledgements that need conversation context."""
    value = _clean_text(message).casefold().strip(" .!?؟،؛\"'“”‘’")
    # Normalize common Arabic spelling variants without changing full queries.
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    confirmations = {
        "ايوا", "ايوه", "ايوة", "ايوه", "اي", "اه", "اها", "نعم", "اجل",
        "تمام", "اكيد", "أكيد", "صح", "بالضبط", "موافق", "موافقه",
        "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "correct",
    }
    return value in confirmations


def _latest_assistant_message(history: str) -> str:
    """Return the most recent assistant turn from normalized chat history."""
    if not history:
        return ""

    assistant_messages = []
    current_role = None
    current_parts = []

    def flush():
        if current_role == "assistant" and current_parts:
            assistant_messages.append(" ".join(current_parts).strip())

    for raw_line in history.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(assistant|user|system|bot|ai|human|customer)\s*:\s*(.*)$", line, re.I)
        if match:
            flush()
            role = match.group(1).lower()
            current_role = "assistant" if role in {"assistant", "bot", "ai"} else role
            current_parts = [match.group(2).strip()] if match.group(2).strip() else []
        elif current_role and line:
            current_parts.append(line)

    flush()
    return assistant_messages[-1] if assistant_messages else ""


def _latest_user_message(history: str) -> str:
    """Return the most recent user turn from role-labelled conversation history."""
    if not history:
        return ""

    user_messages = []
    current_role = None
    current_parts = []

    def flush():
        if current_role == "user" and current_parts:
            user_messages.append(" ".join(current_parts).strip())

    for raw_line in history.splitlines():
        line = raw_line.strip()
        match = re.match(
            r"^(assistant|user|system|bot|ai|human|customer|contact|agent)\s*:\s*(.*)$",
            line,
            re.I,
        )
        if match:
            flush()
            role = match.group(1).lower()
            current_role = "user" if role in {"user", "human", "customer", "contact"} else role
            current_parts = [match.group(2).strip()] if match.group(2).strip() else []
        elif current_role and line:
            current_parts.append(line)

    flush()
    return user_messages[-1] if user_messages else ""


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

        # Each search is an independent DB round trip: run them in parallel.
        futures = [
            search_executor.submit(
                search_knowledge,
                embedding,
                client_id=client_id,
                limit=SEARCH_LIMIT,
                query_text=query_text,
            )
            for query_text, embedding in zip(texts, embeddings)
        ]

        return [future.result() for future in futures]

    except Exception:
        logger.exception("Knowledge search failed: client_id=%s", client_id)
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
            logger.warning("Rerank skipped: VOYAGE_API_KEY is not configured")
            return results

        voyage_client = voyageai.Client(api_key=api_key)
        reranked = voyage_client.rerank(
            query=query,
            documents=[_clean_content(result.content) for result in results],
            model="rerank-2",
            top_k=min(CONTEXT_LIMIT, len(results)),
        )

        return [results[item.index] for item in reranked.results]

    except Exception:
        logger.exception("Rerank failed; using retrieval order")
        return results


def _answer_key(content: str) -> str:
    """The fact an entry states: its answer for Q&A entries, else the content."""
    content = _clean_content(content)

    if "Answer:" in content:
        content = content.split("Answer:", 1)[1]

    return re.sub(r"\s+", " ", content).strip().casefold()


def _unique_answers(results):
    """
    Keep the best-ranked entry per distinct answer.

    The knowledge holds many rephrasings of one question with the same
    answer; without this they fill every context slot with a single fact
    and push other relevant facts (e.g. a fee) out of the context.
    """
    seen: set[str] = set()
    unique = []

    for result in results:
        key = _answer_key(result.content)

        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

    return unique


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
    resolved_question: str = "",
):
    """Let Claude answer from retrieved knowledge instead of pre-rejecting it."""
    answer = generate_answer(
        question=question,
        context=context,
        client_config=client_config,
        history=history,
        resolved_question=resolved_question,
    )

    if not isinstance(answer, str) or not answer.strip():
        logger.warning("Grounded answer empty; falling back to knowledge-gap handling")
        return None

    return {
        "answer": _clean_answer_format(answer),
        "best_distance": best_distance,
        "status": "KNOWN",
    }


# --------------------------------------------------
# Main RAG pipeline
# --------------------------------------------------


def _history(conversation_context: str | None) -> str:
    return trim_history(conversation_context, MAX_HISTORY_CHARS)


# A business writes its handoff policy as knowledge entries that start with
# one of these labels, e.g. "سياسة التحويل: طلبات إلغاء المواعيد تحول للموظف".
# Only these entries steer routing; other entries that merely mention a
# transfer ("سيتم تحويلها للقسم المختص") do not.
HANDOFF_POLICY_LABELS = ("سياسة التحويل", "handoff policy")
POLICY_LABEL_WINDOW = 40  # the label must appear at the start of the entry
MAX_HANDOFF_POLICIES = 30


def is_handoff_policy(content: str) -> bool:
    head = (content or "").strip()[:POLICY_LABEL_WINDOW].casefold()
    return any(label in head for label in HANDOFF_POLICY_LABELS)


def handoff_policies(client_id: int) -> str:
    """All of this client's handoff policy entries (a few lines, no search)."""
    entries = [
        content
        for content in get_handoff_policy_candidates(
            client_id,
            HANDOFF_POLICY_LABELS,
            POLICY_LABEL_WINDOW,
            MAX_HANDOFF_POLICIES,
        )
        if is_handoff_policy(content)
    ]

    return "\n".join(f"- {' '.join(entry.split())}" for entry in entries)


def build_search_query(
    question: str,
    client_config: dict | None = None,
    conversation_context: str | None = None,
) -> str:
    """
    Turn the latest message into a standalone search query (or CASUAL_MARKER).

    The chat route calls this after intent classification and passes
    the resolved query into the RAG pipeline.
    """
    cleaned_question = _clean_text(question)

    if not cleaned_question:
        return cleaned_question

    # Claude resolves references using history and creates a standalone search query.
    search_query = rewrite_query(
        cleaned_question,
        client_config=client_config,
        history=_history(conversation_context),
    )

    if not isinstance(search_query, str) or not search_query.strip():
        return cleaned_question

    return search_query.strip()


def answer_question(
    question: str,
    client_id: int,
    client_config: dict | None = None,
    conversation_context: str | None = None,
    search_query: str | None = None,
):
    """
    Multi-tenant RAG pipeline with conversation-aware retrieval.

    The latest user message is interpreted in context. Knowledge retrieval
    remains isolated by client_id.

    search_query: optional result of build_search_query() computed ahead of
    time; when omitted it is computed here.
    """
    logger.info(
        "RAG request: client_id=%s config_loaded=%s context_chars=%s",
        client_id,
        bool(client_config),
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
    history = _history(conversation_context)

    if not search_query:
        search_query = build_search_query(
            question,
            client_config=client_config,
            conversation_context=conversation_context,
        )

    # Customer text: DEBUG only, so it stays out of normal production logs.
    logger.debug("Original query: %s", cleaned_question)
    logger.debug("Search query: %s", search_query)

    # Keep retrieval context-aware even when the rewriter misses a reference.
    # Search terms are generated from the previous user turn + previous assistant
    # turn + current message, so follow-ups such as "كم سعره؟", "وين موقعكم؟",
    # or "طيب كيف أحجز؟" retain the topic without hardcoding specific phrases.
    previous_user = _latest_user_message(history)
    previous_assistant = _latest_assistant_message(history)
    contextual_retrieval_query = " ".join(
        part for part in (previous_user, previous_assistant, cleaned_question) if part
    ).strip()

    # If the rewriter mistakenly labels a contextual confirmation as casual,
    # do not discard the business context. Use the recent exchange for retrieval.
    if search_query == CASUAL_MARKER and _is_short_confirmation(cleaned_question):
        if previous_user or previous_assistant:
            logger.info("Contextual reply fallback: using recent exchange for retrieval")
            search_query = contextual_retrieval_query or previous_assistant or previous_user

    # Handle ordinary casual conversation only when it is not a contextual
    # short confirmation (or no preceding assistant turn is available).
    if search_query == CASUAL_MARKER:
        casual_answer = generate_casual_answer(
            cleaned_question,
            client_config=client_config,
            history=history,
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

    # Always add a contextual retrieval candidate when there is a previous turn.
    # The direct query remains in the list, so a clearly new topic can still win;
    # tenant isolation is preserved by _search_many(client_id).
    if contextual_retrieval_query and contextual_retrieval_query.casefold() not in {
        search_query.casefold(), cleaned_question.casefold()
    }:
        texts.append(contextual_retrieval_query)

    if (
        cleaned_question
        and cleaned_question.casefold() != search_query.casefold()
        and cleaned_question.casefold() not in {item.casefold() for item in texts}
    ):
        texts.append(cleaned_question)

    result_lists = _search_many(texts, client_id)
    search_failed = result_lists is None

    if search_failed:
        logger.warning("Retrieval unavailable; continuing with client configuration")
        result_lists = []

    results = _unique_answers([
        result
        for result in _merge(*result_lists)
        if (
            float(result.distance) <= MAX_DISTANCE
            or int(getattr(result, "keyword_hits", 0) or 0) > 0
        )
        and not is_handoff_policy(result.content)
    ])

    if results:
        results = _rerank(search_query, results)
        results = results[:CONTEXT_LIMIT]

    logger.info(
        "Retrieved %s knowledge entries: ids=%s",
        len(results),
        [result.id for result in results],
    )

    for result in results:
        logger.debug(
            "Result id=%s distance=%s keyword_hits=%s rank_score=%s content=%s",
            result.id,
            result.distance,
            getattr(result, "keyword_hits", 0),
            getattr(result, "rank_score", result.distance),
            result.content,
        )

    knowledge_context = _build_context(results)

    best_distance = min(
        (float(result.distance) for result in results),
        default=None,
    )

    # Evaluate question completeness and evidence together after retrieval.
    # This prevents a named-but-unknown subject from being mistaken for an
    # incomplete question, and avoids generating an answer that will be discarded.
    evaluation = evaluate_question(
        question=search_query or cleaned_question,
        latest_message=cleaned_question,
        context=knowledge_context,
        client_config=client_config,
        history=history,
    )

    question_status = evaluation["question_status"]
    evidence_status = evaluation["evidence_status"]

    logger.info(
        "Question evaluation: client_id=%s question_status=%s "
        "evidence_status=%s result_count=%s",
        client_id,
        question_status,
        evidence_status,
        len(results),
    )

    if question_status == "INCOMPLETE":
        clarification = evaluation["clarification_question"] or (
            "يرجى تحديد العنصر الذي تقصده حتى أتمكن من مساعدتك."
            if re.search(r"[\u0600-\u06FF]", cleaned_question)
            else "Please specify which item you mean so I can help."
        )
        return {
            "answer": _clean_answer_format(clarification),
            "best_distance": best_distance,
            "status": "CLARIFICATION_REQUIRED",
        }

    if evidence_status == "NOT_ANSWERABLE":
        gap_type = classify_knowledge_gap(
            question=search_query or cleaned_question,
            context=knowledge_context,
            client_config=client_config,
            history=history,
        )

        logger.info(
            "Knowledge gap: client_id=%s type=%s",
            client_id,
            gap_type,
        )

        fallback = generate_fallback_answer(
            question=cleaned_question,
            gap_type=gap_type,
            client_config=client_config,
        )

        return {
            "answer": _clean_answer_format(fallback),
            "best_distance": best_distance,
            "status": "SEARCH_ERROR" if search_failed else gap_type,
        }

    grounded_result = _generate_grounded_answer(
        question=cleaned_question,
        context=knowledge_context,
        client_config=client_config,
        best_distance=best_distance,
        history=history,
        resolved_question=search_query,
    )

    if grounded_result:
        if search_failed or not results:
            grounded_result["status"] = "CONFIG_ONLY"

        return grounded_result

    gap_type = classify_knowledge_gap(
        question=search_query or cleaned_question,
        context=knowledge_context,
        client_config=client_config,
        history=history,
    )

    logger.info("Knowledge gap: client_id=%s type=%s", client_id, gap_type)

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
