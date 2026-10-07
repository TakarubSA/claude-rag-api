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
Determine whether the user's question is related to the subject,
services, or domain described by the provided context.

Return ONLY one of these two values:

RELATED_BUT_UNKNOWN
OUT_OF_SCOPE

RELATED_BUT_UNKNOWN means:
The question is related to the subject or services described
in the context, but the context does not contain enough
information to answer it.

OUT_OF_SCOPE means:
The question is unrelated to the subject, services, or domain
described in the context.

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