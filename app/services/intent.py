import os
import re
import logging

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# --------------------------------------------------
# Text normalization
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize Arabic text and digits for matching."""
    text = (text or "").lower()

    # Convert Arabic and Persian digits to English digits.
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


# --------------------------------------------------
# Delta Laboratories booking flow
# --------------------------------------------------

LAB_MARKERS = (
    "دلتا",
    "delta",
    "مختبر",
    "مختبرات",
    "المختبر",
    "تحاليل",
    "تحليل",
    "فحص",
    "فحوصات",
    "laboratory",
    "laboratories",
    "lab",
)

BOOKING_MARKERS = (
    "رقم الحجز",
    "رقم حجز",
    "رقم حجزي",
    "رقم الحجوز",
    "حجز التحاليل",
    "حجز تحاليل",
    "booking number",
    "booking reference",
    "booking no",
    "booking id",
    "reservation number",
    "my booking",
)

DETAIL_REQUEST_MARKERS = (
    "اسم صاحب التحليل",
    "اسم التحليل",
    "قائمة التحاليل المطلوبة",
)


def is_lab_booking_request(
    message: str,
    conversation_context: str = "",
) -> bool:
    """
    Detect a laboratory booking-number workflow.

    Pass recent conversation messages as conversation_context.
    Do not pass the knowledge-base documents as conversation history.
    """
    msg = normalize_text(message)
    context = normalize_text(conversation_context)
    combined = f"{context}\n{msg}"

    has_lab_reference = any(
        marker in combined for marker in LAB_MARKERS
    )

    has_booking_reference = any(
        marker in combined for marker in BOOKING_MARKERS
    )

    # Catch common variants such as "رقم الحجز" and "رقم حجزي".
    if not has_booking_reference:
        has_booking_reference = (
            "رقم" in combined
            and any(
                word in combined
                for word in ("حجز", "الحجز", "حجزي", "حجوز")
            )
        )

    # Detect a conversation where the bot already asked
    # for the patient's name, age, and test.
    bot_requested_details = (
        any(marker in context for marker in DETAIL_REQUEST_MARKERS)
        and any(marker in context for marker in LAB_MARKERS)
        and (
            any(marker in context for marker in BOOKING_MARKERS)
            or "اصدار رقم الحجز" in context
        )
    )

    return (
        has_lab_reference and has_booking_reference
    ) or bot_requested_details


def has_lab_booking_details(message: str) -> bool:
    """Check for patient name, age, and test in one or multiple lines."""
    text = normalize_text(message)

    if not text:
        return False

    # Find age, accepting Arabic or English digits.
    age_match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", text)
    if not age_match:
        return False

    age = int(age_match.group(1))
    if not 1 <= age <= 120:
        return False

    # Require a lab test or analysis.
    test_markers = (
        "تحليل", "تحاليل", "فحص", "فحوصات",
        "اشعه", "test", "tests", "analysis",
    )
    if not any(marker in text for marker in test_markers):
        return False

    ignored_words = {
        "اسم", "صاحب", "التحليل", "العمر", "عمري",
        "سنه", "سنوات", "عام", "اعوام", "رقم",
        "الحجز", "حجز", "من", "فضلا", "الرجاء",
        "تحليل", "تحاليل", "فحص", "فحوصات",
        "السكر", "سكر", "الضغط", "ضغط", "الدهون",
        "الدم", "الحمل", "دلتا", "مختبر", "مختبرات",
        "اشعه", "test", "tests", "analysis",
        "blood", "sugar", "pressure", "patient",
        "name", "age",
    }

    words = re.findall(r"[\u0600-\u06FFA-Za-z]+", text)
    name_tokens = [
        word for word in words
        if word not in ignored_words
    ]

    return bool(name_tokens)

# --------------------------------------------------
# Intent classifier
# --------------------------------------------------

def classify_intent(
    message: str,
    conversation_context: str = "",
) -> str:
    """
    Return exactly one action:
    reply, api, or handoff.

    Delta booking behavior:
    - Booking-number request without all required details -> reply.
    - The customer provides the details in an active booking flow
      -> handoff.
    """

    # Rule 1: Handle Delta/lab booking requests deterministically.
    # This check runs before Claude so that Claude cannot override it.
    if is_lab_booking_request(message, conversation_context):
        if has_lab_booking_details(message):
            logger.info(
                "Intent classified as handoff: lab booking details received."
            )
            return "handoff"

        logger.info(
            "Intent classified as reply: lab booking details are needed."
        )
        return "reply"

    # Rule 2: Classify all other intents using Claude.
    prompt = f"""
Classify the user's message into exactly ONE of these actions:

reply
api
handoff

Definitions:

reply:
Use this when the user is asking for general information that can be
answered using the knowledge base, or when the user is greeting
or starting a conversation.

Examples:
- كيف أحجز موعد؟
- ايش طرق الدفع؟
- وين موقع العيادة؟
- هل أقدر أغير موعدي؟
- السلام عليكم
- مرحبا
- صباح الخير
- كيف حالك؟

api:
Use this when the user wants the system to perform an operation,
check live data, retrieve specific information, or interact with
an external system/API.

IMPORTANT:
A request for specific live information may be api even if it is phrased
as a question, unless it is a laboratory booking-number request that
must be routed to a human agent.

Examples:
- هل رحلتي 12345 صار فيها نداء؟
- شيك على رحلتي 12345
- وين طلبي 98765؟
- شيك على موعدي 4567
- هل الطلب 12345 تم شحنه؟
- ابحث عن فاتورتي 12345

handoff:
Use this when the user explicitly or implicitly wants to speak
with a real human agent, customer service representative,
employee, or specialist.

Examples:
- أبغى أكلم موظف
- أريد التحدث مع موظف
- حولني لموظف
- أبغى خدمة العملاء
- ممكن أكلم شخص حقيقي؟
- أريد التحدث مع شخص
- احتاج موظف يساعدني
- حولني لأحد الموظفين
- أبغى أكلم الدعم
- ما أبغى أكمل مع البوت

Priority rules:
1. Explicit or implicit request for a human agent -> handoff.
2. Requests for live or external data -> api.
3. Otherwise -> reply.

Recent conversation context:
{conversation_context or "(none supplied)"}

Current user message:
{message}

Return ONLY one of:
reply
api
handoff
"""

    try:
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

                logger.warning(
                    "Unexpected Claude classification: %s",
                    result,
                )
                return "reply"

        logger.warning("Claude returned no text block.")
        return "reply"

    except Exception:
        logger.exception("Intent classification failed.")
        return "reply"