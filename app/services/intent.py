import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def classify_intent(message: str):
    prompt = f"""
Classify the user's message into exactly one of these categories:

QUESTION
ACTION
GREETING
OTHER

Definitions:

QUESTION:
The user is asking for general information that can be answered
from the knowledge base.

Examples:
- كيف أحجز موعد؟
- ايش طرق الدفع؟
- وين موقع العيادة؟
- هل أقدر أغير موعدي؟

ACTION:
The user wants the system to perform an operation, check live data,
retrieve specific information, or interact with an external system.

IMPORTANT:
A message can be written as a question and still be ACTION.

If the user asks about a specific:
- flight number
- booking number
- order number
- appointment number
- customer number
- tracking number
- or any other identifier

and the system needs to check information about it, classify it as ACTION.

Examples:
- هل رحلتي 12345 صار فيها نداء؟
- شيك على رحلتي 12345
- أبغى أعرف حالة الرحلة 12345
- وين طلبي 98765؟
- شيك على موعدي 4567
- هل الطلب 12345 تم شحنه؟

GREETING:
The user is greeting or starting a conversation.

Examples:
- السلام عليكم
- هلا
- مرحبا
- صباح الخير

OTHER:
Anything that does not fit the categories above.

User message:
{message}

Return ONLY the category name.
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=100,
    thinking={
    "type": "between_tools"
},
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    for block in response.content:
        if hasattr(block, "text") and block.text:
            return block.text.strip()

    print("CLAUDE RESPONSE:", response)

    raise RuntimeError("Claude did not return a text response")