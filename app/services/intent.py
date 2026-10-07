import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def classify_intent(message: str):
    prompt = f"""
Classify the user's message into exactly ONE of these actions:

reply
api
handoff

Definitions:

reply:
Use this when the user is asking for general information that can be
answered using the knowledge base, or when the user is simply greeting
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
A message can be written as a question and still be api.

If the user asks about a specific:
- flight number
- booking number
- order number
- appointment number
- customer number
- tracking number
- invoice number
- or any other identifier

and the system needs to check or retrieve information about it,
use api.

Examples:
- هل رحلتي 12345 صار فيها نداء؟
- شيك على رحلتي 12345
- أبغى أعرف حالة الرحلة 12345
- وين طلبي 98765؟
- شيك على موعدي 4567
- هل الطلب 12345 تم شحنه؟
- ابحث عن فاتورتي 12345

handoff:
Use this when the user explicitly or implicitly wants to speak
with a real human agent, customer service representative, employee,
or specialist.

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
- أبغى شخص من خدمة العملاء

IMPORTANT:
If the user requests a human agent, always use handoff even if
the message also contains a question, complaint, or problem.

Examples:
- ما قدرت أحجز وأبغى أكلم موظف
- عندي مشكلة في موعدي، حولني لموظف
- الطلب ما وصلني وأبغى شخص يساعدني
- ممكن أحد من خدمة العملاء يتواصل معي؟

Priority rules:

1. If the user wants a human agent → handoff
2. If the user needs live/specific/external data → api
3. Otherwise → reply

User message:
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

            allowed_actions = {
                "reply",
                "api",
                "handoff",
            }

            if result in allowed_actions:
                return result

            print(
                "Unexpected Claude classification:",
                result,
            )

            return "reply"

    # Claude returned no text block.
    # Don't crash the /chat endpoint.
    print(
        "Claude returned no text block.",
        "Content:",
        response.content,
    )

    return "reply"
