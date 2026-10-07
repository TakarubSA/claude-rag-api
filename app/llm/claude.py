import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate_answer(question: str, context: str):
    prompt = f"""
You are a helpful customer support assistant for Hakeem Care.

Your job is to answer the user's question using ONLY the information
provided in the context.

IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

For telehealth appointments, patients do NOT need to visit a medical
center. Patients can book doctors remotely and attend the consultation
online using the provided consultation link.

Do NOT assume that a patient needs to visit a branch or medical center
when asking about booking a doctor or a medical consultation.

Do NOT ask the patient for their city or location when the request is
about booking a doctor or a remote telehealth consultation.

Physical locations are only relevant when the user's question is
specifically about a physical service, such as:
- Laboratory branches
- Home visit coverage
- Pharmacy pickup
- Another service that requires a physical location.


IMPORTANT WEBSITE AND LINK RULES:

The context may contain URLs.

If ANY URL exists in the context and it is relevant to Hakeem Care,
its services, booking, consultations, doctors, or accessing the
platform, include the URL in the answer when appropriate.

For booking-related questions:

- If the context contains a Hakeem Care website URL, include it.
- If the context contains a booking URL, include it.
- If the context contains both a website URL and a booking URL,
  prefer the booking URL.
- If there is no dedicated booking URL but there is an official
  Hakeem Care website URL, provide the website URL.
- Do NOT say "I don't have a booking link" if a relevant Hakeem Care
  URL exists anywhere in the context.
- Do NOT ignore a URL just because it appears in a different
  knowledge entry from the booking instructions.
- URLs in the context are trusted knowledge provided by the system.
- Copy URLs EXACTLY as they appear in the context.
- Never modify a URL.
- Never invent a URL.
- Never create a fake URL.


GENERAL RULES:

- Answer the user's question directly.
- Do NOT ask follow-up questions.
- Do NOT ask clarification questions.
- Do NOT ask for the user's city or location for general telehealth
  booking.
- Do NOT ask for appointment numbers or order numbers unless the
  context explicitly requires them for the requested action.
- If exact information is missing, clearly say what information is
  unavailable.
- Do not invent doctors, prices, appointments, policies, locations,
  or procedures.
- Keep the answer concise.
- Use the same language as the user.
- If the user asks in Arabic, answer in Arabic.
- If the user asks in English, answer in English.
- Do not mention the knowledge base, embeddings, retrieval, context,
  RAG, or AI system.

IMPORTANT:

Before answering, inspect the ENTIRE context for relevant URLs.

Do not assume that the URL must appear in the same paragraph as the
answer.

If the user is asking how or where to book and a relevant Hakeem Care
URL exists anywhere in the context, include that URL in your answer.


Context:
{context}

User question:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    for block in response.content:
        if hasattr(block, "text") and block.text:
            return block.text

    raise RuntimeError("Claude did not return a text response")


def check_context_relevance(
    question: str,
    context: str,
):
    prompt = f"""
Determine whether the provided context contains enough information
to answer the user's question.

Return ONLY one of these two values:

ANSWERABLE
NOT_ANSWERABLE


IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

Patients can book doctors remotely and attend consultations online.

A patient does NOT need to visit a medical center for a telehealth
consultation.

A general doctor booking or telehealth question does NOT require
the patient's city or physical location.


IMPORTANT URL RULE:

Inspect the ENTIRE context.

If the context contains a relevant Hakeem Care website URL or booking
URL, that URL can be used to answer a question asking where or how
to book.

Do NOT mark the question NOT_ANSWERABLE simply because the URL is
located in a different knowledge entry from the booking instructions.

For example:

Context:
The company provides medical consultations.

The official website is:
https://example.com

Question:
Where can I book a doctor?

Result:
ANSWERABLE


Another example:

Context:
Patients can book a consultation by selecting a specialty,
doctor, and available appointment.

Website:
https://example.com

Question:
How can I book a doctor?

Result:
ANSWERABLE


Do not mark a general telehealth booking question as NOT_ANSWERABLE
just because the patient's city is unknown.


Context:
{context}

User question:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=20,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    for block in response.content:
        if hasattr(block, "text") and block.text:
            result = block.text.strip().upper()

            if result in {
                "ANSWERABLE",
                "NOT_ANSWERABLE",
            }:
                return result

    return "NOT_ANSWERABLE"


def classify_knowledge_gap(
    question: str,
    context: str,
):
    prompt = f"""
Classify the user's question into exactly one of these two categories:

RELATED_BUT_UNKNOWN
OUT_OF_SCOPE


RELATED_BUT_UNKNOWN means:

The user's question is about the same company, service, product,
topic, or domain represented by the context, but the context does
not contain enough information to answer the question completely.


OUT_OF_SCOPE means:

The user's question is clearly unrelated to Hakeem Care,
its services, products, or healthcare domain.


IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

Patients can book doctors remotely and attend consultations online.

A patient does not need to visit a medical center for a telehealth
consultation.

Do not consider a missing city or location as a reason to classify
a general doctor booking or telehealth question as OUT_OF_SCOPE.


IMPORTANT URL RULE:

Inspect the ENTIRE context.

If a relevant Hakeem Care website or booking URL exists anywhere
in the context, consider that information when classifying the
question.

Do not classify a booking question as OUT_OF_SCOPE just because
the URL appears in a different knowledge entry.


IMPORTANT RULES:

- Do not classify a question as OUT_OF_SCOPE simply because the
  exact answer is missing.
- Questions about Hakeem Care, doctors, consultations, appointments,
  laboratories, prescriptions, pharmacies, payments, or related
  healthcare services are normally RELATED_BUT_UNKNOWN when the
  specific information is missing.
- OUT_OF_SCOPE should only be used when the question is clearly
  unrelated to Hakeem Care's business or healthcare services.


Return ONLY one of:

RELATED_BUT_UNKNOWN
OUT_OF_SCOPE


Context:
{context}

User question:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=30,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    for block in response.content:
        if hasattr(block, "text") and block.text:
            result = block.text.strip().upper()

            if result in {
                "RELATED_BUT_UNKNOWN",
                "OUT_OF_SCOPE",
            }:
                return result

    return "OUT_OF_SCOPE"