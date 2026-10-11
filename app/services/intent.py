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


VALID_INTENTS = {"reply", "api", "handoff"}


def classify_intent(
    message: str,
    conversation_context: str = "",
) -> str:
    """
    Classify the user's current intent using the message and context.

    Valid actions:
    - reply
    - api
    - handoff

    The classifier considers conversation history but does not
    automatically carry an old handoff decision into new messages.
    """

    message = (message or "").strip()

    conversation_context = (
        conversation_context or ""
    ).strip()[-8000:]

    if not message:
        return "reply"

    prompt = f"""
You are an intent classifier for a multi-tenant customer-support system.

Your task is to identify the user's CURRENT intent using the latest
message and relevant conversation history.

Return exactly one action:
reply
api
handoff

IMPORTANT PRINCIPLES

1. Classify what the user wants NOW, not what they wanted earlier.
2. Use conversation history to understand meaning and references.
3. Do not assume an old request or handoff is still active merely
   because it appears in the conversation history.
4. Do not classify based only on individual keywords.
5. Understand Arabic, Saudi Arabic dialect, English, and mixed-language
   messages according to their meaning and context.
6. If there is insufficient evidence for a handoff, choose reply,
   unless the user clearly requests an external operation, in which
   case choose api.

RECENT CONVERSATION:
<conversation>
{conversation_context or "(No conversation history supplied)"}
</conversation>

LATEST USER MESSAGE:
<latest_message>
{message}
</latest_message>

ACTION DEFINITIONS

reply:
Choose reply when the user is having a normal conversation, asking
a general question, greeting the assistant, answering an ordinary
clarification question, thanking the assistant, or changing the topic.

Examples:
- سلام
- السلام عليكم
- هلا والله
- كيف حالك؟
- مين أنتم؟
- كيف أحجز موعد؟
- وين موقعكم؟
- شكراً
- The assistant asks which city the user means, and the user answers.
- The user previously discussed customer support but now asks a
  general question unrelated to that request.

These examples illustrate intent, not a fixed keyword list.
Determine the meaning of the actual message.

api:
Choose api when the user requests an operation or live/external
information that the system should retrieve or perform.

Examples:
- Check the status of my order.
- شيك على حالة طلبي.
- هل عندي موعد بكرة؟
- أبغى أتابع الشحنة.
- Retrieve a specific booking or transaction status.

Choose api only when the requested operation is appropriate for
an external API and no applicable mandatory handoff rule overrides it.

handoff:
Choose handoff only when there is clear evidence that human support
should take over NOW.

Valid reasons:

A. EXPLICIT HUMAN REQUEST

The latest message clearly asks to speak with, contact, or be
transferred to a human employee, support agent, or specialist.

B. ACTIVE HUMAN-SUPPORT WORKFLOW

The recent conversation establishes that the assistant requested
specific information to complete or investigate a human-support request.

The latest user message provides the requested information or clearly
confirms the next step, and that information is sufficient to proceed.

For condition B, verify all of the following:
- The previous assistant message was specifically about human support.
- The request has not already been completed or become irrelevant.
- The latest user message responds meaningfully to that request.
- The latest message provides the required information or confirmation.

CONTEXT RULES

- A previous mention of an employee does not itself justify handoff.
- A previous handoff decision does not automatically apply to the next turn.
- A greeting does not become a handoff because an employee was mentioned
  earlier in the conversation.
- A new question must be classified according to its current meaning.
- An answer to an ordinary clarification question is reply.
- Providing a name, age, phone number, or other detail is not sufficient
  for handoff unless the active conversation establishes why that detail
  was requested.
- If the user supplies information requested for an active human-support
  workflow, do not ask them to repeat information already supplied.
- If the user explicitly requests a human now, choose handoff even if
  the conversation previously discussed another topic.
- Do not invent a human-support workflow that is not supported by context.
- Do not confuse an assistant's suggestion to contact support with the
  user's actual decision to proceed.
- If the latest message is ambiguous and there is no clear evidence that
  a transfer should happen now, prefer reply.

EXAMPLES

Example 1:
Earlier conversation: The user asked to speak to customer support.
Latest user message: سلام
Action: reply

Example 2:
Earlier conversation: The assistant discussed an employee.
Latest user message: كيف أحجز موعد؟
Action: reply

Example 3:
Assistant: Please provide your name so our support team can investigate.
User: Ammar
Action: handoff

Example 4:
Assistant: Which city do you mean?
User: Jeddah
Action: reply

Example 5:
User: أبغى أكلم موظف
Action: handoff

Example 6:
Assistant: Would you like me to connect you with a human agent?
User: لا، عندي سؤال ثاني
Action: reply

Example 7:
Earlier conversation: A support request was completed.
Latest user message: شكراً
Action: reply

Example 8:
User: شيك على حالة الطلب رقم 12345
Action: api

PRIORITY

1. Clear, explicit request for a human now -> handoff.
2. Clear response to an active human-support workflow -> handoff.
3. External operation or live information request -> api,
   unless a mandatory handoff rule applies.
4. Otherwise -> reply.

The conversation history provides context; it does not override the
actual meaning of the latest message without a relevant reason.

Return exactly one word:
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

        response_text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip().lower()

        # Accept a valid action if Claude adds simple punctuation
        # or formatting around the answer.
        normalized_result = (
            response_text
            .replace("`", "")
            .strip()
            .rstrip(" .!:\n")
        )

        if normalized_result in VALID_INTENTS:
            logger.info(
                "Intent classified: intent=%s",
                normalized_result,
            )
            return normalized_result

        # Fail safely if the model returns an unexpected response.
        # Do not accidentally trigger a handoff from malformed output.
        logger.warning(
            "Unexpected intent classification: %r",
            response_text,
        )
        return "reply"

    except Exception:
        logger.exception("Intent classification failed.")
        return "reply"
