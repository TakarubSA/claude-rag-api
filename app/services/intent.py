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
    """
    Accept booking details in either format:

    عمار ٢٨ تحليل البول

    or:

    عمار
    ٢٨
    تحليل البول
    """
    text = normalize_text(message)

    # Find the age, accepting Arabic and English digits.
    age_match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", text)

    if not age_match:
        logger.info(
            "Booking details incomplete: age not found. text=%r",
            text,
        )
        return False

    age = int(age_match.group(1))

    if not 1 <= age <= 120:
        logger.info(
            "Booking details incomplete: invalid age=%s",
            age,
        )
        return False

    # Require a patient name before the age.
    name_part = text[:age_match.start()].strip()

    ignored_words = {
        "اسم", "صاحب", "التحليل", "تحليل", "تحاليل",
        "العمر", "عمري", "عمر", "سنة", "سنه",
        "سنوات", "عام", "اعوام", "المريض", "المريضة",
    }

    name_words = re.findall(
        r"[ء-يA-Za-z]+",
        name_part,
    )

    patient_name = [
        word for word in name_words
        if word not in ignored_words
    ]

    if not patient_name:
        logger.info(
            "Booking details incomplete: patient name not found. text=%r",
            text,
        )
        return False

    # Require an analysis/test name.
    test_markers = (
        "تحليل", "تحاليل", "فحص", "فحوصات",
        "السكر", "سكر", "البول", "بول", "ابول",
        "الضغط", "ضغط", "الدم", "الدهون",
        "cbc", "urine", "glucose", "sugar",
        "test", "tests", "analysis",
    )

    if not any(marker in text for marker in test_markers):
        logger.info(
            "Booking details incomplete: test not found. text=%r",
            text,
        )
        return False

    logger.info(
        "Booking details complete: name=%s age=%s",
        " ".join(patient_name),
        age,
    )
    return True
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
Classify the user's latest message into exactly ONE action:

reply
api
handoff

You will receive:
- Recent conversation context, which may include previous bot messages.
- The user's latest message.

IMPORTANT GENERAL HANDOFF RULE:

If the previous conversation shows that the assistant asked the
user to provide specific information, details, documents, or answers,
and the latest user message appears to provide the requested
information, classify the latest message as:

handoff

This rule applies to ALL clients and ALL business domains.
Do not hardcode any particular client, company, service, or laboratory.

Examples:

Previous assistant message:
"Please provide your full name, age, and test name."

Latest user message:
"Ammar, 28 years old, blood test."

Action:
handoff

Previous assistant message:
"Please send your order number so our team can investigate."

Latest user message:
"Order 12345."

Action:
handoff

Previous assistant message:
"Please provide your name and phone number so customer service
can follow up."

Latest user message:
"My name is Ahmed, my phone number is ..."

Action:
handoff

The latest user message may contain the requested information
on one line, across multiple lines, or in a different order.
Understand Arabic and English, including Arabic-Indic digits.

Do not ask for the information again once the user has provided it.
Do not classify a message as handoff merely because it contains
a name, number, or other information. The conversation context
must show that the assistant previously requested that information,
unless the user independently requests a human agent.

Other classifications:

reply:
General questions answerable using the knowledge base, greetings,
and ordinary conversation. Also use reply when the assistant needs
to ask the user for missing information.

api:
Requests requiring an external API or live data retrieval,
unless one of the handoff rules above applies.

handoff:
The user explicitly requests a human agent, customer service,
employee, or specialist.

Priority rules:

1. User has provided information requested by the assistant
   in the previous conversation -> handoff.
2. User explicitly or implicitly requests a human agent -> handoff.
3. User asks for live data or an external operation -> api.
4. Otherwise -> reply.

Recent conversation context:
{conversation_context or "(no previous conversation context provided)"}

Latest user message:
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