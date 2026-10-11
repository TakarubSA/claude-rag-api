from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.services.intent import (
    classify_intent,
    has_lab_booking_details,
    is_lab_booking_request,
)
from app.services.rag import answer_question


router = APIRouter()

# Hakeem Care tenant
HAKEEM_CLIENT_ID = 1


class ChatRequest(BaseModel):
    message: str
    conversation_context: str = ""


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
    # 1. Validate Authorization header
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

    # 2. Extract token
    token = authorization[len("Bearer "):].strip()

    # 3. Resolve token -> client
    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    message = request.message.strip()
    context = request.conversation_context or ""

    # 4. Handle Hakeem Care Delta lab booking requests
    # before RAG so retrieval cannot override this workflow.
    if client.id == HAKEEM_CLIENT_ID:

        # Customer provided name, age, and test.
        # Transfer directly to an agent.
        if has_lab_booking_details(message):
            print(
                "INTENT: handoff | REASON: lab booking details received"
            )

            return chat_response(
                action="handoff",
                answer=(
                    "شكرًا لك، استلمنا البيانات المطلوبة 🌷\n\n"
                    "سأحوّل محادثتك الآن إلى فريق خدمة العملاء "
                    "لاستكمال إصدار رقم الحجز. يرجى الانتظار قليلًا."
                ),
            )

        # Customer asks for a lab booking number but hasn't
        # supplied the required information.
        if is_lab_booking_request(message, context):
            print(
                "INTENT: reply | REASON: lab booking details needed"
            )

            return chat_response(
                action="reply",
                answer=(
                    "أبشر، نساعدك في الحصول على رقم الحجز "
                    "لتحاليل مختبرات دلتا 🌷\n\n"
                    "فضلاً أرسل لنا المعلومات التالية:\n"
                    "1. اسم صاحب التحليل.\n"
                    "2. العمر.\n"
                    "3. اسم التحليل أو قائمة التحاليل المطلوبة.\n\n"
                    "بعد استلام البيانات، سنحوّل محادثتك إلى "
                    "فريق خدمة العملاء لاستكمال إصدار رقم الحجز."
                ),
            )

    # 5. Classify other requests.
    # Pass context so follow-up messages can be interpreted correctly.
    intent = classify_intent(
        message,
        conversation_context=context,
    )

    print("CLIENT ID:", client.id)
    print("INTENT:", intent)

    # 6. Answer using this client's knowledge base.
    result = answer_question(
        question=message,
        client_id=client.id,
        client_config=client.ai_config,
    )

    # 7. Save knowledge gaps.
    if result["status"] == "RELATED_BUT_UNKNOWN":
        insert_gap(
            client_id=client.id,
            message=message,
            gap_type=result["status"],
        )

    # 8. Return response.
    return {
        "action": intent,
        "answer": result["answer"],
        "best_distance": result["best_distance"],
        "status": result["status"],
    }