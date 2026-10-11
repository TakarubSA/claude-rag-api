import logging
import os

import anthropic
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv(
    "ANTHROPIC_INTENT_MODEL",
    os.getenv(
        "ANTHROPIC_MODEL",
        "claude-haiku-4-5-20251001",
    ),
)

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# Greetings that should never trigger a handoff on their own.
GREETINGS = {
    "سلام",
    "السلام عليكم",
    "السلام عليكم ورحمة الله",
    "السلام عليكم ورحمة الله وبركاته",
    "هلا",
    "هلا والله",
    "مرحبا",
    "مرحباً",
    "مرحبا بك",
    "أهلا",
    "أهلاً",
    "اهلا",
    "اهلاً",
    "صباح الخير",
    "مساء الخير",
    "مساء النور",
    "صباح النور",
    "hi",
    "hello",
    "hey",
}


def classify_intent(
    message: str,
    conversation_context: str = "",
) -> str:
    """
    Classify the user's latest message.

    Valid actions:
    - reply
    - api
    - handoff

    Conversation context is used to recognize answers to
    previous clarification questions and human handoff workflows.

    A standalone greeting always returns reply, regardless
    of previous conversation history.
    """

    message = (message or "").strip()

    conversation_context = (
        conversation_context or ""
    ).strip()[-8000:]

    if not message:
        return "reply"

    # Normalize whitespace and common Arabic character variants.
    normalized_message = " ".join(message.lower().split())

    # Standalone greeting must not trigger a handoff because
    # an earlier message in the conversation mentioned an agent.
    if normalized_message in GREETINGS:
        logger.info(
            "Intent shortcut: standalone greeting -> reply"
        )
        return "reply"

    prompt = f"""
You are an intent classifier for a multi-tenant customer-support system.

Classify the user's latest message into exactly ONE action:

reply
api
handoff

Use the recent conversation to understand what the user is doing now.
Consider the conversation history when it is relevant, but classify
the user's latest message according to its actual meaning.

RECENT CONVERSATION:
{conversation_context or "(no conversation history supplied)"}

LATEST USER MESSAGE:
{message}

ACTION DEFINITIONS

reply:
Use when the user asks a general question that can be answered from
the business knowledge base, greets the assistant, or provides an answer
to an ordinary clarification question so the assistant can continue
answering the user's question.

Examples:
- كيف أحجز موعد؟
- ما طرق الدفع؟
- وين موقع العيادة؟
- هل هو دكتور مختص؟
- The assistant asks which city the user means, and the user says Riyadh.
- The user says سلام, مرحبا, هلا, or hello without another request.

api:
Use when the user asks the system to perform an operation or retrieve
live, specific, external information, and no handoff rule applies.

Examples:
- Check the status of order 12345.
- هل الطلب 12345 تم شحنه؟
- شيك على موعدي.

handoff:
Use when either of these conditions is met:

1. The user explicitly or implicitly asks to speak with a human employee,
   support agent, customer service representative, or specialist.

2. The recent conversation shows that the assistant specifically asked
   the user to provide information needed for a human team to handle,
   complete, investigate, issue, or follow up on a request, and the latest
   user message supplies the requested information sufficiently to proceed.

For condition 2:
- Check the previous assistant messages, not only the latest message.
- The user may send details on one line or across multiple lines.
- Details may be in Arabic or English.
- Arabic-Indic digits and English digits both count.
- The fields can be given in any order.
- Do not ask for the same details again once they have been provided.
- Do not use handoff merely because the user gives a name, age, number,
  or other information.
- A simple answer to a normal clarification question is not handoff unless
  the prior context indicates that the information is for a human follow-up
  or a transfer workflow.
- Do not classify a standalone greeting as handoff merely because
  an earlier conversation mentioned an agent or human support.
- A previous handoff request does not automatically make every later
  message a handoff.

Examples:

Context:
Assistant: Please provide your name, age, and test name so customer
service can complete your request.
Latest user message:
عمار
٢٨ سنة
تحليل البول
Action: handoff

Context:
Assistant: Send your order number so our support team can investigate.
Latest user message:
Order 12345
Action: handoff

Context:
Assistant: Which city are you asking about?
Latest user message:
Riyadh
Action: reply

Context:
Assistant: Would you like to talk to a human agent?
Latest user message:
Yes, please.
Action: handoff

Context:
Earlier conversation discussed contacting customer support.
Latest user message:
سلام
Action: reply

PRIORITY:
1. Standalone greeting -> reply.
2. Explicit request for a human agent -> handoff.
3. Requested information for human handling is sufficiently supplied -> handoff.
4. Live/external operation -> api.
5. Otherwise -> reply.

Return only one word:
reply
api
handoff
"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        for block in response.content:
            if getattr(block, "type", None) != "text":
                continue

            result = (block.text or "").strip().lower()

            # Accept a valid action even if the model adds punctuation
            # or wraps it in simple formatting.
            result = result.replace("`", "").strip()
            result = result.rstrip(" .!:\n")

            if result in {"reply", "api", "handoff"}:
                return result

            # If Claude returns an unexpected response, fail safely.
            logger.warning(
                "Unexpected intent classification: %r",
                result,
            )
            return "reply"

        logger.warning("Intent classifier returned no text.")
        return "reply"

    except Exception:
        logger.exception("Intent classification failed.")
        return "reply"
