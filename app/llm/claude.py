import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate_answer(question: str, context: str):
    prompt = f"""
You are a helpful customer support assistant.

Answer the user's question using only the information provided in the context.

If the answer is not available in the context, say that you don't have enough information.

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
        if hasattr(block, "text"):
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

For example:

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

For example:

Context:
The company provides healthcare services.

Question:
How many ants are there in the world?

Result:
OUT_OF_SCOPE

Important rules:
- Do not classify a question as OUT_OF_SCOPE simply because
  the exact answer is missing from the context.
- If the question mentions or asks about the company, its services,
  products, doctors, locations, prices, appointments, or related
  business information, it should normally be RELATED_BUT_UNKNOWN
  when the context does not provide the answer.
- OUT_OF_SCOPE should only be used when the question is clearly
  unrelated to the company's domain.

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