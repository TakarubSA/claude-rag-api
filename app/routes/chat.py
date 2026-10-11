import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict
from fastapi import APIRouter, Header, HTTPException, Request
from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.services.intent import classify_intent
from app.services.rag import answer_question


router = APIRouter()
logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 8000
MAX_CONTEXT_MESSAGES = 20

# Common names used by chat integrations to carry conversation history.
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

# These fields may identify a conversation but are not history themselves.
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
    # Preserve additional fields instead of silently discarding them.
    model_config = ConfigDict(extra="allow")

    message: str
    conversation_context: Any = None


def _extract_content(value: Any) -> str:
    """Extract readable text from strings, dictionaries, and content arrays."""
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
                part = (
                    item.get("text")
                    or item.get("content")
                    or item.get("message")
                )
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
        # Handle common wrappers around conversation history.
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
            if key in value and value[key] is not None:
                extracted = _extract_content(value[key])
                if extracted:
                    return extracted

        content = (
            value.get("content")
            or value.get("message")
            or value.get("text")
        )

        if content is not None:
            return _extract_content(content)

        return ""

    return ""


def _format_single_message(item: Any) -> str:
    """Format one conversation message as role: content."""
    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return ""

    content = _extract_content(
        item.get("content")
        or item.get("message")
        or item.get("text")
    )

    if not content:
        return ""

    role = (
        item.get("role")
        or item.get("sender")
        or item.get("author")
        or item.get("type")
        or "message"
    )

    role = str(role).strip().lower()

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
    """
    Normalize conversation history into readable text.

    Supports:
    - Plain strings
    - Lists of message dictionaries
    - Dictionaries containing messages/history/turns
    - OpenAI-style content arrays
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()[-MAX_CONTEXT_CHARS:]

    if isinstance(value, dict):
        # If this is a single message, format it directly.
        line = _format_single_message(value)
        if line:
            return line[-MAX_CONTEXT_CHARS:]

        # Otherwise, try to extract a nested history structure.
        extracted = _extract_content(value)
        return extracted[-MAX_CONTEXT_CHARS:] if extracted else ""

    if isinstance(value, list):
        lines = []

        for item in value[-MAX_CONTEXT_MESSAGES:]:
            line = _format_single_message(item)

            if line:
                lines.append(line)

        return "\n".join(lines)[-MAX_CONTEXT_CHARS:]

    return str(value).strip()[-MAX_CONTEXT_CHARS:]


def get_conversation_context(
    request: ChatRequest,
) -> tuple[str, str]:
    """
    Find usable conversation history in the request.

    Known history fields are checked first. Additional fields are checked
    afterward when their names suggest that they contain conversation history.
    """
    payload = request.model_dump(exclude={"message"})

    # Prefer the most explicit history field names.
    ordered_names = list(CONTEXT_FIELD_NAMES)

    # Include additional fields with likely history-related names.
    for key in payload:
        normalized_key = key.lower()

        if any(
            marker in normalized_key
            for marker in (
                "conversation",
                "history",
                "messages",
                "transcript",
                "chat_context",
                "chat_history",
            )
        ):
            if key not in ordered_names:
                ordered_names.append(key)

    for field_name in ordered_names:
        if field_name not in payload:
            continue

        context = format_conversation_context(payload[field_name])

        if context:
            return context, field_name

    return "", "none"


def get_request_diagnostics(
    request: ChatRequest,
) -> dict[str, list[str]]:
    """
    Return field names useful for debugging the ChatGate integration.
    Does not expose field values or message contents.
    """
    payload = request.model_dump(exclude={"message"})
    extra_fields = request.model_extra or {}

    possible_context_fields = []
    possible_identifier_fields = []

    for key in payload:
        normalized_key = key.lower()

        if any(
            marker in normalized_key
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

        if any(
            marker in normalized_key
            for marker in CONVERSATION_ID_MARKERS
        ):
            possible_identifier_fields.append(key)

    return {
        "payload_fields": sorted(payload.keys()),
        "extra_fields": sorted(extra_fields.keys()),
        "possible_context_fields": sorted(possible_context_fields),
        "possible_conversation_id_fields": sorted(
            possible_identifier_fields
        ),
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


@router.post("/chat")
def chat(
    request: ChatRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
):
    logger.info("HEADERS: %s", {k: v for k, v in http_request.headers.items() if k.lower() != "authorization"})
    logger.info("BODY: %s", request.model_dump())
    # 1. Validate Authorization.
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

    # 2. Resolve token to the client.
    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    message = request.message.strip()

    if not message:
        return make_response(
            action="reply",
            answer="فضلاً اكتب رسالتك حتى أتمكن من مساعدتك.",
            status="EMPTY_QUESTION",
        )

    # 3. Extract conversation history and diagnose the incoming payload.
    conversation_context, context_source = get_conversation_context(request)
    diagnostics = get_request_diagnostics(request)

    logger.info(
        (
            "Chat request received: client_id=%s "
            "context_received=%s context_chars=%s context_source=%s"
        ),
        client.id,
        bool(conversation_context),
        len(conversation_context),
        context_source,
    )

    logger.info(
        "Incoming request field names: %s",
        diagnostics["payload_fields"],
    )

    logger.info(
        "Incoming extra field names: %s",
        diagnostics["extra_fields"],
    )

    logger.info(
        "Possible conversation history fields: %s",
        diagnostics["possible_context_fields"],
    )

    logger.info(
        "Possible conversation identifier fields: %s",
        diagnostics["possible_conversation_id_fields"],
    )

    # 4. Classify the latest message using conversation history.
    intent = classify_intent(
        message=message,
        conversation_context=conversation_context,
    )

    logger.info(
        "Intent classified: client_id=%s intent=%s",
        client.id,
        intent,
    )

    # 5. Handoff responses must bypass RAG.
    if intent == "handoff":
        return make_response(
            action="handoff",
            answer=(
                "شكرًا لك، استلمنا المعلومات المطلوبة. "
                "سيتم تحويل محادثتك إلى فريق خدمة العملاء "
                "لمتابعة طلبك. يرجى الانتظار."
            ),
        )

    # 6. Answer using this client's isolated knowledge base and context.
    result = answer_question(
        question=message,
        client_id=client.id,
        client_config=client.ai_config,
        conversation_context=conversation_context,
    )

    # 7. Record knowledge gaps.
    if result.get("status") == "RELATED_BUT_UNKNOWN":
        insert_gap(
            client_id=client.id,
            message=message,
            gap_type=result["status"],
        )

    # 8. Return response.
    return make_response(
        action=intent,
        answer=result["answer"],
        status=result["status"],
        best_distance=result.get("best_distance"),
    )