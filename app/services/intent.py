import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def normalize_text(text: str) -> str:
    """Normalize Arabic and English text for reliable keyword matching."""
    text = (text or "").lower()

    # Normalize Arabic-Indic and Persian digits.
    text = text.translate(
        str.maketrans(
            "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
            "01234567890123456789",
        )
    )

    # Remove Arabic diacritics and tatweel.
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)

    # Normalize common Arabic letter variants.
    text = text.translate(
        str.maketrans({
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
        })
    )

    return re.sub(r"\s+", " ", text).strip()


def is_lab_booking_handoff(
    message: str,
    conversation_context: str = "",
) -> bool:
    """
    Route laboratory booking-number requests directly to an agent.

    conversation_context should contain recent conversation messages,
    not the entire knowledge-base context.
    """
    text = normalize_text(
        f"{conversation_context}\n{message}"
    )

    lab_terms = (
        "دلتا",
        "delta",
        "مختبر",
        "مختبرات",
        "المختبر",
        "تحاليل",
        "تحليل",
        "laboratory",
        "laboratories",
        "lab",
    )

    booking_terms = (
        "رقم الحجز",
        "رقم حجز",
        "رقم الحجوز",
        "حجز التحاليل",
        "حجز تحاليل",
        "booking number",
        "booking reference",
        "booking no",
        "booking id",
        "reservation number",
        "booking code",
    )

    has_lab_context = any(term in text for term in lab_terms)
    asks_for_booking_number = any(
        term in text for term in booking_terms
    )

    return has_lab_context and asks_for_booking_number


def classify_intent(
    message: str,
    conversation_context: str = "",
) -> str:
    # Rule 1: Lab booking-number requests always go to an agent.
    # Do not let the model classify these requests as "api".
    if is_lab_booking_handoff(message, conversation_context):
        return "handoff"

    prompt = f"""
Classify the user's message into exactly ONE action:

reply
api
handoff

IMPORTANT BUSINESS RULE — Hakeem Care / Delta Laboratories:

Requests for laboratory booking numbers must be routed to handoff
when the message or recent conversation context indicates that the
request concerns laboratory tests or Delta Laboratories.

Examples:
- ابغى رقم الحجز انا في المختبر
- ما وصلني رقم حجز التحاليل
- حجزت تحاليل دلتا وابغى الرقم
- وين رقم حجز مختبرات دلتا؟
- I need my lab booking number.
- I am at the lab and haven't received my booking number.

For these requests, return handoff, NOT api.
A staff member will collect the patient's name, age, and test name,
then handle issuing the booking number.
Do not ask the user to provide those details before handoff.

reply:
Use for general information answerable from the knowledge base,
greetings, and general conversation.

api:
Use when the system must perform an operation or retrieve specific
live information from an external system/API, except for the
laboratory booking-number workflow defined above.

handoff:
Use when the user explicitly or implicitly needs a human agent,
customer service representative, employee, or specialist.

Examples:
- أبغى أكلم موظف
- حولني لموظف
- أبغى خدمة العملاء
- ممكن أكلم شخص حقيقي؟
- عندي مشكلة وأحتاج موظف يساعدني

Priority rules:
1. Delta/laboratory booking-number request -> handoff.
2. Explicit or implicit request for a human agent -> handoff.
3. Other requests requiring live or external data -> api.
4. Otherwise -> reply.

Recent conversation context:
{conversation_context or "(none supplied)"}

Current user message:
{message}

Return ONLY one of:
reply
api
handoff
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    for block in response.content:
        if hasattr(block, "text") and block.text:
            result = block.text.strip().lower()

            if result in {"reply", "api", "handoff"}:
                return result

            print("Unexpected Claude classification:", result)
            return "reply"

    print("Claude returned no text block.")
    return "reply"