import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.services.intent import classify_intent
from app.services.rag import answer_question


router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    conversation_context: Any = None


def format_conversation_context(value: Any) -> str:
    """
    Normalize conversation history into readable text.

    Supports:
    - plain strings
    - lists of message dictionaries
    - dictionaries containing messages/history/turns
    - OpenAI-style content arrays
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value[-8000:]

    if isinstance(value, dict):
        for key in ("messages", "history", "turns"):
            if key in value:
                return format_conversation_context(value[key])

        role = (
            value.get("role")
            or value.get("sender")
            or value.get("type")
            or "message"
        )

        content = (
            value.get("content")
            or value.get("message")
            or value.get("text")
            or ""
        )

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(
                        str(
                            part.get("text")
                            or part.get("content")
                            or ""
                        )
                    )
                else:
                    parts.append(str(part))

            content = " ".join(parts)

        return f"{role}: {content}"[-8000:]

    if isinstance(value, list):
        lines = []

        for item in value[-20:]:
            if isinstance(item, dict):
                role = (
                    item.get("role")
                    or item.get("sender")
                    or item.get("type")
                    or "message"
                )

                content = (
                    item.get("content")
                    or item.get("message")
                    or item.get("text")
                    or ""
                )

                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(
                                str(
                                    part.get("text")
                                    or part.get("content")
                                    or ""
                                )
                            )
                        else:
                            parts.append(str(part))

                    content = " ".join(parts)

                lines.append(f"{role}: {content}")

            else:
                lines.append(str(item))

        return "\n".join(lines)[-8000:]

    return str(value)[-8000:]


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
    authorization: str | None = Header(default=None),
):
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

    # 3. Normalize context received from ChatGate.
    conversation_context = format_conversation_context(
        request.conversation_context
    )

    logger.info(
        "Chat request received: client_id=%s context_received=%s context_chars=%s",
        client.id,
        bool(conversation_context),
        len(conversation_context),
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

    # 6. Answer using this client's isolated knowledge base.
    result = answer_question(
        question=message,
        client_id=client.id,
        client_config=client.ai_config,
        conversation_context=conversation_context,
    )

    # 7. Record knowledge gaps.
    if result["status"] == "RELATED_BUT_UNKNOWN":
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
        best_distance=result["best_distance"],
    )