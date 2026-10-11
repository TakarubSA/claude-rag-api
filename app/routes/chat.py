from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.services.intent import classify_intent
from app.services.rag import answer_question


router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_context: str | None = None


def chat_response(
    action: str,
    answer: str,
    status: str = "KNOWN",
):
    return {
        "action": action,
        "answer": answer,
        "best_distance": None,
        "status": status,
    }


@router.post("/chat")
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    # 1. Validate Authorization
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

    # 2. Resolve token to client
    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    message = request.message.strip()
    context = request.conversation_context or ""

    # 3. Classify intent using the current message and
    # recent conversation context.
    intent = classify_intent(
        message,
        conversation_context=context,
    )

    print("CLIENT ID:", client.id)
    print("INTENT:", intent)

    # 4. Handoff should not go through RAG.
    if intent == "handoff":
        print("HANDOFF: requested or collected information received")

        return chat_response(
            action="handoff",
            answer=(
                "شكرًا لك، استلمنا المعلومات المطلوبة. "
                "سنحوّل محادثتك إلى فريق خدمة العملاء "
                "لاستكمال طلبك. يرجى الانتظار قليلًا."
            ),
        )

    # 5. Answer other requests using this client's knowledge base.
    result = answer_question(
        question=message,
        client_id=client.id,
        client_config=client.ai_config,
    )

    # 6. Save knowledge gaps.
    if result["status"] == "RELATED_BUT_UNKNOWN":
        insert_gap(
            client_id=client.id,
            message=message,
            gap_type=result["status"],
        )

    # 7. Return response.
    return {
        "action": intent,
        "answer": result["answer"],
        "best_distance": result["best_distance"],
        "status": result["status"],
    }