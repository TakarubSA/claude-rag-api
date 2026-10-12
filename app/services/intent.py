
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

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

VALID_INTENTS = {"reply", "api", "handoff"}


def classify_intent(
    message: str,
    conversation_context: str = "",
) -> str:
    """
    Classify the user's current intent using Claude and recent context.

    Returns exactly one action:
    - reply: Answer using the knowledge base or normal conversation.
    - api: Execute a requested external or live operation.
    - handoff: Transfer to a human agent when justified.
    """
    message = (message or "").strip()
    conversation_context = (
        conversation_context or ""
    ).strip()[-8000:]

    if not message:
        return "reply"

    prompt = f"""
You are an intent classifier for a multi-tenant
customer-support system.

Determine the user's CURRENT intent from the latest message
and relevant conversation history.

Return exactly one action:
reply, api, or handoff.

GENERAL RULES

- Understand Arabic, Saudi Arabic dialect, English,
  and mixed-language messages.
- Classify the user's latest message, not an earlier request.
- Use conversation history to understand references and ongoing
  workflows, without blindly repeating previous decisions.
- Interpret the user's meaning instead of relying on isolated
  words or keywords.
- When the intent is ambiguous, prefer reply unless an external
  operation or human transfer is clearly required.
- A request for information about a service is not, by itself,
  a request to execute an operation.

ACTIONS

reply:
Use reply for normal conversation and questions that can be
answered using the business knowledge base or available
customer-support information.

This includes:
- Questions about services, policies, prices, and procedures.
- Requests for information or guidance.
- Questions about health services and available consultations.
- Descriptions of a problem or symptoms when the user is
  asking for information, guidance, or a suitable service.
- Questions about how to book or use a service.
- Greetings, acknowledgements, clarifications, and topic changes.
- Requests that can be answered without executing an external
  operation or transferring the conversation to a human.

For medical-service inquiries, classify the intent based on
what the user is asking the assistant to do.
A description of symptoms or a request for medical guidance
does not, on its own, indicate that the user wants a human agent.

api:
Use api only when the user is requesting an actual external
or live operation that the system needs to execute, such as
checking a live order status, retrieving a transaction status,
or performing a supported booking operation.

Do not use api merely because the user asks how a service works,
asks about its price, or wants information about availability.

Only select api when the requested operation is supported
by the system.

handoff:
Use handoff only when there is clear evidence that the
conversation needs to be transferred to a human agent.

A handoff is appropriate when:
1. The user explicitly requests a human agent or asks to speak
   with a customer-support employee.
2. An active human-support workflow requires information or
   confirmation, the assistant clearly requested it to proceed,
   and the user's latest message supplies that information
   or confirms the required step.
3. A known business rule explicitly requires a human agent
   for the user's current request.

HANDOFF SAFEGUARDS

- Do not infer that the user wants a human merely because
  their issue is important, complicated, urgent, or personal.
- Do not automatically transfer a conversation because the
  user describes a problem or requests assistance.
- Do not treat a request for advice, service information,
  pricing, or booking instructions as consent to a handoff.
- Do not invent an active human-support workflow.
- Mentioning an employee earlier does not automatically
  require a handoff.
- A previous handoff decision does not automatically apply
  to the current message.
- Providing a name, age, phone number, or other information
  does not independently justify a handoff.
- A reply to an ordinary clarification question should remain
  reply unless the conversation clearly establishes an active
  human-support workflow.
- If the user changes the subject, classify the new request
  independently.
- Do not select handoff simply because the assistant previously
  suggested speaking with an employee.
- Explicitly requesting a human agent now takes precedence
  over unrelated earlier topics.
- Use the knowledge base and available business rules to
  determine whether a mandatory handoff applies.

PRIORITY

1. An explicit request to speak with a human agent:
   handoff.
2. A clearly active human-support workflow requiring transfer:
   handoff.
3. A supported external operation explicitly requested by
   the user: api.
4. Otherwise: reply.

RECENT CONVERSATION:
<conversation>
{conversation_context or "(No conversation history supplied)"}
</conversation>

LATEST USER MESSAGE:
<latest_message>
{message}
</latest_message>

Return exactly one word:
reply, api, or handoff.
""".strip()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,
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

        normalized = (
            response_text
            .replace("`", "")
            .strip()
            .rstrip(" .!:\n")
        )

        if normalized in VALID_INTENTS:
            logger.info(
                "Intent classified: intent=%s",
                normalized,
            )
            return normalized

        logger.warning(
            "Unexpected intent classification: %r",
            response_text,
        )
        return "reply"

    except Exception:
        logger.exception("Intent classification failed.")
        return "reply"
