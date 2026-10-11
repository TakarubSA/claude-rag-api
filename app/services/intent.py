import logging
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv(
    "ANTHROPIC_INTENT_MODEL",
    os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

VALID_INTENTS = {"reply", "api", "handoff"}


def classify_intent(message: str, conversation_context: str = "") -> str:
    """Classify the user's current intent using Claude and recent context."""
    message = (message or "").strip()
    conversation_context = (conversation_context or "").strip()[-8000:]

    if not message:
        return "reply"

    prompt = f"""
You are an intent classifier for a multi-tenant customer-support system.

Identify the user's CURRENT intent from the latest message and relevant history.
Return exactly one action: reply, api, or handoff.

GENERAL RULES
- Understand Arabic, Saudi Arabic dialect, English, and mixed-language messages.
- Classify the latest message, not an earlier request.
- Use history to resolve references and active workflows, but don't blindly carry
  a previous handoff decision into a new turn.
- Do not classify based on isolated keywords.
- If evidence for handoff is ambiguous, choose reply unless an external operation
  is clearly requested, in which case choose api.

ACTIONS
reply:
Normal conversation, questions answerable from knowledge, greetings, thanks,
ordinary clarification replies, or topic changes.

api:
The user asks the system to perform an external/live operation such as checking
an order, booking, transaction, or shipment status. Use only when the request
actually requires an external operation and no mandatory handoff rule applies.

handoff:
1. The latest message explicitly asks to speak to or be transferred to a human.
2. The conversation has an active human-support workflow: the assistant recently
   asked for specific information/confirmation to proceed, the workflow is still
   active, and the latest message supplies that information or confirms the step.

HANDOFF SAFEGUARDS
- Mentioning an employee earlier is not enough.
- A prior handoff result does not automatically apply to this turn.
- A name, age, phone number, or other detail alone does not trigger handoff unless
  the recent conversation clearly shows it was requested for an active support case.
- A response to an ordinary clarification question is reply.
- Do not invent a support workflow or treat the assistant's suggestion as user consent.
- Explicitly asking for a human now overrides unrelated prior topics.

EXAMPLES
- Earlier asked for support; latest message is "سلام" -> reply
- Assistant asks which city; user says "جدة" -> reply
- Assistant asks for name to investigate an active support request; user gives name -> handoff
- User says "أبغى أكلم موظف" -> handoff
- User says "لا، عندي سؤال ثاني" after being offered a human -> reply
- User says "شيك على حالة الطلب رقم 12345" -> api

PRIORITY
1. Explicit request for a human now -> handoff
2. Clear response advancing an active human-support workflow -> handoff
3. External/live operation -> api
4. Otherwise -> reply

RECENT CONVERSATION:
<conversation>
{conversation_context or "(No conversation history supplied)"}
</conversation>

LATEST USER MESSAGE:
<latest_message>
{message}
</latest_message>

Return exactly one word: reply, api, handoff
""".strip()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip().lower()
        normalized = response_text.replace("`", "").strip().rstrip(" .!:\n")

        if normalized in VALID_INTENTS:
            logger.info("Intent classified: intent=%s", normalized)
            return normalized

        logger.warning("Unexpected intent classification: %r", response_text)
        return "reply"
    except Exception:
        logger.exception("Intent classification failed.")
        return "reply"
