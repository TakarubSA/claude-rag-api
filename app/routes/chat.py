import contextvars
import logging
import re
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.llm.usage import use_client
from app.services.history import drop_current_message, trim_history
from app.services.conversation_memory import (
    clear_conversation,
    get_history,
    save_turn,
)
from app.services.intent import classify_intent
from app.services.rag import (
    answer_question,
    build_search_query,
    executor,
    knowledge_for_intent,
)


router = APIRouter()
logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 8000

# A chat message longer than this is not a real question; it only costs
# money (every message triggers several Claude calls).
MAX_MESSAGE_CHARS = 2000
MAX_CONTEXT_MESSAGES = 20

CONTEXT_FIELD_NAMES = (
    "conversation_context",
    "conversation_history",
    "chat_history",
    "history",
    "messages",
    "transcript",
    "chat_context",
    "context",
)

CONVERSATION_ID_MARKERS = (
    "conversation_id",
    "conversationid",
    "session_id",
    "sessionid",
    "thread_id",
    "threadid",
    "chat_id",
    "chatid",
)


class ChatRequest(BaseModel):
    # Preserve integration-specific fields for diagnostics/history extraction.
    model_config = ConfigDict(extra="allow")

    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    conversation_context: Any = None
    conversation_id: str | None = None
    reset_conversation: bool = False


def _is_identifier_field(field_name: str) -> bool:
    normalized = field_name.lower()
    return any(marker in normalized for marker in CONVERSATION_ID_MARKERS)


def _is_control_field(field_name: str) -> bool:
    return field_name.lower() in {"reset_conversation"}


def _extract_content(value: Any) -> str:
    """Extract readable text from common message/content formats."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                part = item.get("text") or item.get("content") or item.get("message")
                if part is not None:
                    extracted = _extract_content(part)
                    if extracted:
                        parts.append(extracted)
            else:
                extracted = _extract_content(item)
                if extracted:
                    parts.append(extracted)
        return " ".join(parts).strip()

    if isinstance(value, dict):
        for key in (
            "messages",
            "history",
            "turns",
            "conversation_history",
            "conversation_context",
            "chat_history",
            "transcript",
            "data",
            "payload",
        ):
            nested = value.get(key)
            if nested is not None:
                extracted = format_conversation_context(nested)
                if extracted:
                    return extracted

        content = value.get("content") or value.get("message") or value.get("text")
        if content is not None:
            return _extract_content(content)

    return ""


def _format_single_message(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return ""

    content = _extract_content(
        item.get("content") or item.get("message") or item.get("text")
    )
    if not content:
        return ""

    role = str(
        item.get("role")
        or item.get("sender")
        or item.get("author")
        or item.get("type")
        or "message"
    ).strip().lower()

    role_aliases = {
        "human": "user",
        "customer": "user",
        "contact": "user",
        "bot": "assistant",
        "ai": "assistant",
        "agent": "assistant",
    }
    role = role_aliases.get(role, role)
    return f"{role}: {content}"


def format_conversation_context(value: Any) -> str:
    """Normalize plain text, lists, and wrapped message histories."""
    if value is None:
        return ""

    if isinstance(value, str):
        return trim_history(value, MAX_CONTEXT_CHARS)

    if isinstance(value, list):
        lines = []
        for item in value[-MAX_CONTEXT_MESSAGES:]:
            line = _format_single_message(item)
            if line:
                lines.append(line)
            elif not isinstance(item, dict):
                text = _extract_content(item)
                if text:
                    lines.append(text)
        return trim_history("\n".join(lines), MAX_CONTEXT_CHARS)

    if isinstance(value, dict):
        for key in (
            "messages",
            "history",
            "turns",
            "conversation_history",
            "conversation_context",
            "chat_history",
            "transcript",
        ):
            if key in value and value[key] is not None:
                nested = format_conversation_context(value[key])
                if nested:
                    return nested

        line = _format_single_message(value)
        if line:
            return trim_history(line, MAX_CONTEXT_CHARS)

        extracted = _extract_content(value)
        return trim_history(extracted, MAX_CONTEXT_CHARS)

    return trim_history(str(value), MAX_CONTEXT_CHARS)


def get_conversation_context(request: ChatRequest) -> tuple[str, str]:
    payload = request.model_dump(exclude={"message"})
    candidates = list(CONTEXT_FIELD_NAMES)

    for key, value in payload.items():
        normalized = key.lower()
        if (
            value is not None
            and not _is_identifier_field(key)
            and not _is_control_field(key)
            and any(
                marker in normalized
                for marker in (
                    "conversation",
                    "history",
                    "messages",
                    "transcript",
                    "chat_context",
                    "chat_history",
                )
            )
            and key not in candidates
        ):
            candidates.append(key)

    for field_name in candidates:
        if _is_identifier_field(field_name) or _is_control_field(field_name):
            continue

        value = payload.get(field_name)
        if value is None:
            continue

        context = format_conversation_context(value)
        if context:
            return context, field_name

    return "", "none"


def get_request_diagnostics(request: ChatRequest) -> dict[str, list[str]]:
    provided_fields = sorted(request.model_fields_set)
    extra_fields = sorted((request.model_extra or {}).keys())
    possible_context_fields = []
    possible_identifier_fields = []

    for key in provided_fields:
        normalized = key.lower()
        if _is_identifier_field(key):
            possible_identifier_fields.append(key)
        elif not _is_control_field(key) and any(
            marker in normalized
            for marker in (
                "conversation",
                "history",
                "messages",
                "transcript",
                "chat_context",
                "chat_history",
            )
        ):
            possible_context_fields.append(key)

    return {
        "payload_fields": provided_fields,
        "extra_fields": extra_fields,
        "possible_context_fields": sorted(possible_context_fields),
        "possible_conversation_id_fields": sorted(possible_identifier_fields),
    }


def make_response(
    action: str,
    answer: str,
    status: str = "KNOWN",
    best_distance=None,
):
    return {
        "action": action,
        "answer": answer,
        "best_distance": best_distance,
        "status": status,
    }


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _handoff_answer(message: str) -> str:
    """Generic handoff text, used when the classifier did not write one."""
    if _is_arabic(message):
        return (
            "شكرًا لك، استلمنا طلبك. "
            "سيتم تحويل محادثتك إلى فريق خدمة العملاء "
            "لمتابعة طلبك. يرجى الانتظار."
        )

    return (
        "Thank you, we have received your request. "
        "Your conversation is being transferred to our "
        "customer service team. Please wait."
    )


def _log_timing(client_id: int, path: str, started: float) -> None:
    logger.info(
        "Chat answered: client_id=%s path=%s total_ms=%.0f",
        client_id,
        path,
        (time.perf_counter() - started) * 1000,
    )


@router.post("/chat")
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is required",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format",
        )

    token = authorization[len("Bearer "):].strip()
    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Claude calls for this request use the provider's key and are billed to it.
    use_client(client.id, client.anthropic_api_key)

    started = time.perf_counter()
    message = request.message.strip()

    if not message:
        return make_response(
            action="reply",
            answer="فضلاً اكتب رسالتك حتى أتمكن من مساعدتك.",
            status="EMPTY_QUESTION",
        )

    conversation_id = (request.conversation_id or "").strip()

    if len(conversation_id) > 200:
        raise HTTPException(
            status_code=400,
            detail="conversation_id must be 200 characters or fewer",
        )

    if request.reset_conversation and conversation_id:
        clear_conversation(client.id, conversation_id)

    incoming_context, context_source = get_conversation_context(request)
    stored_context = get_history(client.id, conversation_id) if conversation_id else ""
    # Some integrations send the transcript including the message being
    # asked now; it must not be mistaken for the previous turn.
    conversation_context = drop_current_message(
        incoming_context or stored_context,
        message,
    )

    diagnostics = get_request_diagnostics(request)

    logger.info(
        "Chat request: client_id=%s conversation_id_present=%s context_received=%s "
        "context_chars=%s context_source=%s memory_context=%s",
        client.id,
        bool(conversation_id),
        bool(conversation_context),
        len(conversation_context),
        context_source if incoming_context else (
            "server_memory" if stored_context else "none"
        ),
        bool(stored_context),
    )
    logger.debug("Request diagnostics: %s", diagnostics)

    if not conversation_id:
        logger.warning(
            "No conversation_id provided; temporary memory is disabled for this request."
        )

    # Rewrite the search query while the intent is classified: the two
    # Claude calls are independent. copy_context() keeps this request's
    # provider key and usage tracking in the worker thread.
    search_query_future = executor.submit(
        contextvars.copy_context().run,
        build_search_query,
        message,
        client_config=client.ai_config,
        conversation_context=conversation_context,
    )

    # The routing step reads the client's own knowledge: each business
    # defines there which cases go to staff and how to contact it.
    try:
        intent_knowledge = knowledge_for_intent(
            message,
            client.id,
            conversation_context,
        )
    except Exception:
        logger.exception("Routing knowledge lookup failed: client_id=%s", client.id)
        intent_knowledge = ""

    decision = classify_intent(
        message=message,
        conversation_context=conversation_context,
        knowledge=intent_knowledge,
    )

    intent = decision["action"]

    logger.info(
        "Intent decision: client_id=%s intent=%s action=%s reason=%s",
        client.id,
        decision["intent"],
        decision["action"],
        decision["reason"],
    )

    if decision.get("needs_clarification"):
        clarification_answer = (
            decision.get("clarification_question", "").strip()
            or (
                "أكيد، ممكن توضح لي أكثر ما الذي تحتاج المساعدة فيه؟"
                if _is_arabic(message)
                else "Of course. Could you tell me a little more about what you need help with?"
            )
        )

        save_turn(
            client.id,
            conversation_id,
            message,
            clarification_answer,
        )
        _log_timing(client.id, "clarification", started)

        return make_response(
            action="reply",
            answer=clarification_answer,
            status="CLARIFICATION_REQUIRED",
        )

    if intent == "handoff":
        # The query rewrite, if still running, is simply not used.
        handoff_answer = decision.get("handoff_message") or _handoff_answer(message)
        save_turn(client.id, conversation_id, message, handoff_answer)
        _log_timing(client.id, "handoff", started)
        return make_response(action="handoff", answer=handoff_answer)

    search_query = None

    try:
        search_query = search_query_future.result()
    except Exception:
        # answer_question can still perform its own query preparation.
        logger.exception(
            "Search query preparation failed: client_id=%s",
            client.id,
        )

    result = answer_question(
        question=message,
        client_id=client.id,
        client_config=client.ai_config,
        conversation_context=conversation_context,
        search_query=search_query,
    )

    if result.get("status") == "RELATED_BUT_UNKNOWN":
        # Gap tracking is analytics: never fail the user's answer over it.
        try:
            insert_gap(
                client_id=client.id,
                message=message,
                gap_type=result["status"],
            )
        except Exception:
            logger.exception("Failed to record knowledge gap: client_id=%s", client.id)

    answer = result.get("answer", "")

    if answer:
        save_turn(client.id, conversation_id, message, answer)

    _log_timing(client.id, f"answer/{result.get('status')}", started)

    return make_response(
        action=intent,
        answer=answer,
        status=result.get("status", "UNKNOWN"),
        best_distance=result.get("best_distance"),
    )