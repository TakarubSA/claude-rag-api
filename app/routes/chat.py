import json
import logging
import re

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.clients.repository import get_client_by_token
from app.knowledge.gaps_repository import insert_gap
from app.services.intent import (
    classify_intent,
    is_lab_booking_request,
    normalize_text,
)
from app.services.rag import answer_question


router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    conversation_context: str = ""


# --------------------------------------------------
# Tenant configuration
# --------------------------------------------------

def get_booking_flow_config(client_config) -> dict:
    """
    Read the booking workflow configuration for any tenant.

    Expected ai_config structure:
    {
        "booking_number_flow": {
            "enabled": true
        }
    }
    """
    if isinstance(client_config, str):
        try:
            client_config = json.loads(client_config)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid client ai_config JSON.")
            return {}

    if not isinstance(client_config, dict):
        return {}

    flow_config = client_config.get(
        "booking_number_flow",
        {},
    )

    if not isinstance(flow_config, dict):
        return {}

    return flow_config


# --------------------------------------------------
# Response helper
# --------------------------------------------------

def make_response(
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


# --------------------------------------------------
# Booking-number detection
# --------------------------------------------------

def is_booking_number_request(message: str) -> bool:
    """
    Detect booking-number requests, including messages that
    do not explicitly mention a laboratory.
    """
    text = normalize_text(message)

    booking_phrases = (
        "رقم الحجز",
        "رقم حجز",
        "رقم حجزي",
        "رقم الحجوز",
        "احتاج رقم الحجز",
        "ابغى رقم الحجز",
        "ابي رقم الحجز",
        "اريد رقم الحجز",
        "ما وصلني رقم الحجز",
        "وين رقم الحجز",
        "booking number",
        "booking reference",
        "booking no",
        "booking id",
        "reservation number",
    )

    if any(
        phrase in text
        for phrase in booking_phrases
    ):
        return True

    return (
        "رقم" in text
        and any(
            word in text
            for word in (
                "حجز",
                "الحجز",
                "حجزي",
                "حجوز",
            )
        )
    )


# --------------------------------------------------
# Booking-details detection
# --------------------------------------------------

def has_lab_booking_details(message: str) -> bool:
    """
    Detect all three required details in a single message:
    - Patient name
    - Age
    - Test name

    Supports Arabic digits, English digits, and multiline text.
    """
    text = normalize_text(message)

    # Find the age.
    age_match = re.search(
        r"(?<!\d)(\d{1,3})(?!\d)",
        text,
    )

    if not age_match:
        return False

    age = int(age_match.group(1))

    if not 1 <= age <= 120:
        return False

    # The patient name is expected before the age.
    name_part = text[:age_match.start()].strip()

    name_words = re.findall(
        r"[ء-يA-Za-z]+",
        name_part,
    )

    ignored_name_words = {
        "اسم",
        "صاحب",
        "التحليل",
        "تحليل",
        "تحاليل",
        "العمر",
        "عمري",
        "عمر",
        "سنة",
        "سنوات",
        "سنه",
        "عام",
        "اعوام",
        "الرجاء",
        "فضلا",
        "من",
        "المريض",
        "المريضه",
    }

    patient_name_words = [
        word
        for word in name_words
        if word not in ignored_name_words
    ]

    if not patient_name_words:
        return False

    # Check whether a test or analysis is mentioned.
    test_markers = (
        "تحليل",
        "تحاليل",
        "فحص",
        "فحوصات",
        "السكر",
        "سكر",
        "الضغط",
        "ضغط",
        "البول",
        "بول",
        "ابول",
        "الدم",
        "البراز",
        "cbc",
        "urine",
        "glucose",
        "sugar",
        "pressure",
        "blood test",
        "analysis",
        "test",
        "tests",
    )

    return any(
        marker in text
        for marker in test_markers
    )


# --------------------------------------------------
# Main chat endpoint
# --------------------------------------------------

@router.post("/chat")
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    # 1. Validate Authorization header.
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

    # 2. Extract token.
    token = authorization[len("Bearer "):].strip()

    # 3. Resolve token -> client.
    client = get_client_by_token(token)

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    message = request.message.strip()
    context = request.conversation_context or ""

    # Read workflow settings for this tenant.
    flow_config = get_booking_flow_config(
        client.ai_config
    )

    booking_flow_enabled = (
        flow_config.get("enabled") is True
    )

    logger.info(
        "Chat request received for client_id=%s",
        client.id,
    )

    # --------------------------------------------------
    # 4. Configurable booking-number workflow
    # --------------------------------------------------

    if booking_flow_enabled:

        # First: customer has submitted name, age, and test.
        # Return handoff without calling RAG or Claude.
        if has_lab_booking_details(message):
            logger.info(
                "Booking details detected; returning handoff "
                "for client_id=%s",
                client.id,
            )

            return make_response(
                action="handoff",
                answer=(
                    "شكرًا لك، استلمنا البيانات المطلوبة 🌷\n\n"
                    "سنحوّل محادثتك إلى فريق خدمة العملاء "
                    "لاستكمال إصدار رقم الحجز. يرجى الانتظار."
                ),
            )

        # Second: customer asks for a booking number.
        # Ask for the required details first.
        booking_request = (
            is_booking_number_request(message)
            or is_lab_booking_request(
                message,
                context,
            )
        )

        if booking_request:
            logger.info(
                "Booking details requested from customer; "
                "client_id=%s",
                client.id,
            )

            return make_response(
                action="reply",
                answer=(
                    "أبشر، نساعدك في الحصول على رقم الحجز 🌷\n\n"
                    "فضلاً أرسل لنا البيانات التالية:\n"
                    "1. اسم صاحب التحليل.\n"
                    "2. العمر.\n"
                    "3. اسم التحليل أو قائمة التحاليل المطلوبة.\n\n"
                    "بعد استلام البيانات، سنحوّل محادثتك إلى "
                    "فريق خدمة العملاء لاستكمال إصدار رقم الحجز."
                ),
            )

    # --------------------------------------------------
    # 5. Classify all other requests
    # --------------------------------------------------

    intent = classify_intent(
        message,
        conversation_context=context,
    )

    logger.info(
        "Intent classified as %s for client_id=%s",
        intent,
        client.id,
    )

    # Do not call RAG for handoff responses.
    if intent == "handoff":
        return make_response(
            action="handoff",
            answer=(
                "سأحوّل محادثتك إلى فريق خدمة العملاء "
                "لمساعدتك. يرجى الانتظار."
            ),
        )

    # --------------------------------------------------
    # 6. Answer using this tenant's knowledge base
    # --------------------------------------------------

    result = answer_question(
        question=message,
        client_id=client.id,
        client_config=client.ai_config,
    )

    # --------------------------------------------------
    # 7. Save knowledge gaps
    # --------------------------------------------------

    if result["status"] == "RELATED_BUT_UNKNOWN":
        insert_gap(
            client_id=client.id,
            message=message,
            gap_type=result["status"],
        )

    # --------------------------------------------------
    # 8. Return response
    # --------------------------------------------------

    return {
        "action": intent,
        "answer": result["answer"],
        "best_distance": result["best_distance"],
        "status": result["status"],
    }