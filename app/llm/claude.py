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

Your job is to answer the user's question using only the information
provided in the context.

IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

For telehealth appointments, patients do NOT need to visit a medical
center. A patient can book a doctor remotely and attend the consultation
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
- Another service that actually requires a physical location


GENERAL RULES:

- Answer the user's question directly.
- Do NOT ask follow-up questions.
- Do NOT ask clarification questions.
- Do NOT ask for the user's city, location, appointment number,
  order number, or other information unless the context explicitly
  requires it for the requested action.
- If the exact answer is not available in the context, clearly say
  that the information is not currently available.
- Do not invent or assume information that is not provided.
- Do not make up prices, doctors, locations, policies, or procedures.
- Keep the answer focused and concise.
- Use the same language as the user when possible.
- If the user asks in Arabic, answer in Arabic.
- If the user asks in English, answer in English.
- Do not mention the internal knowledge base, context, embeddings,
  retrieval, or AI system.


IMPORTANT LINK RULES:

- If the context contains a website, booking link, URL, or direct link
  relevant to the user's question, include it in the answer.
- When the user asks how to book, where to book, or how to access a
  service, prefer providing the official website or booking link
  from the context.
- Never invent a URL.
- Never modify a URL.
- Only use URLs exactly as they appear in the context.
- If a relevant URL exists in the context, do not omit it.
- Keep the URL exactly as provided in the context.


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

If the context contains a relevant booking procedure, doctor
information, consultation information, or booking link, consider
the question ANSWERABLE when that information is sufficient to
answer the user's question.

If the context contains a relevant URL or booking link that directly
answers the user's request for where or how to book, consider the
question ANSWERABLE.

Do not mark a question NOT_ANSWERABLE simply because the patient's
city or location is not provided when the question is about remote
telehealth.


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

Important:

If the context contains information about the company, service,
or domain mentioned in the question, consider the question
RELATED_BUT_UNKNOWN even if the specific information requested
is missing.


Example:

Context:
The company provides healthcare services.

Question:
Where is the company located?

Result:
RELATED_BUT_UNKNOWN


Another example:

Context:
The company provides healthcare services.

Question:
What doctors are available today?

Result:
RELATED_BUT_UNKNOWN


OUT_OF_SCOPE means:

The user's question is clearly unrelated to the company,
services, products, or domain represented by the context.


Example:

Context:
The company provides healthcare services.

Question:
How many ants are there in the world?

Result:
OUT_OF_SCOPE


IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

Patients can book doctors remotely and attend consultations online.

A patient does not need to visit a medical center for a telehealth
consultation.

Do not consider a missing city or location as a reason to classify
a general doctor booking or telehealth question as OUT_OF_SCOPE.

If the question is about Hakeem Care, its doctors, consultations,
appointments, laboratories, prescriptions, pharmacies, payments,
or other healthcare services, it should normally be considered
RELATED_BUT_UNKNOWN when the exact information is missing.


IMPORTANT RULES:

- Do not classify a question as OUT_OF_SCOPE simply because
  the exact answer is missing from the context.
- If the question mentions or asks about the company, its services,
  products, doctors, locations, prices, appointments, prescriptions,
  laboratories, pharmacies, payments, or related business
  information, it should normally be RELATED_BUT_UNKNOWN when
  the context does not provide the answer.
- OUT_OF_SCOPE should only be used when the question is clearly
  unrelated to Hakeem Care's business or healthcare services.
- Do not assume that every healthcare question requires a physical
  location.
- General telehealth doctor booking is a remote service.
- A missing city is NOT enough to classify a telehealth request
  as OUT_OF_SCOPE.


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