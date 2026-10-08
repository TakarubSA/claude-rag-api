import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# MODEL = "claude-sonnet-5-5"
MODEL = "claude-haiku-4-5-20251001"

CASUAL_MARKER = "__CASUAL_CONVERSATION__"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _extract_text(response):
    """
    Extract the first actual text block from a Claude response.
    Ignores thinking/tool blocks.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            return block.text.strip()

    return None


def _call_claude(prompt: str, max_tokens: int):
    """
    Single Claude call helper.

    Returns text or None.
    Never raises into the RAG pipeline.
    """
    for attempt in range(2):
        budget = (
            max_tokens
            if attempt == 0
            else max(max_tokens, 1024) + 1500
        )

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=budget,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except Exception as e:
            print(f"[claude_service] API error: {e}")
            return None

        text = _extract_text(response)

        if text:
            return text

        print(
            "[claude_service] no text block "
            f"(stop_reason={getattr(response, 'stop_reason', None)}), retrying"
        )

    return None


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _light_clean(text: str) -> str:
    """
    Light normalization for user input.
    """
    text = (text or "").strip()

    # Arabic diacritics + tatweel
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)

    # هلااااا -> هلاا
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ----------------------------------------------------------------------
# 1) Generic query rewriting
# ----------------------------------------------------------------------

def rewrite_query(question: str) -> str:
    """
    Convert a user's message into a retrieval-friendly search query.

    This function is completely domain-agnostic.
    It does not assume healthcare, products, doctors, or any industry.

    Returns:
        SEARCH_QUERY text
        OR
        __CASUAL_CONVERSATION__
    """

    cleaned = _light_clean(question)

    prompt = f"""
You are a generic query-understanding and search-query rewriting component
for a multi-tenant knowledge assistant.

The assistant serves many different businesses and industries.

Your job is ONLY to understand the user's message and rewrite it into a
better search query for a knowledge base.

The user's message may be:
- Arabic
- English
- Saudi/Gulf/Egyptian/Levantine dialect
- typed with spelling mistakes
- missing letters
- containing Arabizi
- containing merged words
- very short or informal

You must infer the intended meaning from the user's wording.

==================================================
STEP 1 — CASUAL CONVERSATION
==================================================

Return:

CASUAL_CONVERSATION

ONLY when the user is clearly being social and is NOT requesting
information, an action, a product, a service, a policy, a price,
a person, a location, a procedure, or any business-related information.

Examples:

كيفك؟
كيف حالك؟
كيف اخبارك؟
شخبارك؟
ايش مسوي؟
وش تسوي؟
هلا
هلا والله
مرحبا
اهلين
شكرا
مشكور
يعطيكم العافية
الله يعافيك
thanks
hi
hello
باي
مع السلامة

If there is an actual request or question, it is NOT casual.

Examples:

السلام عليكم، ابغى دكتور
السلام عليكم، ابغى سعر المنتج
هلا، وين موقعكم؟
شكراً، بس كم السعر؟

These are SEARCH_QUERY.

==================================================
STEP 2 — SEARCH QUERY
==================================================

Return:

SEARCH_QUERY: <query>

Rules:

1. Preserve the exact meaning of the user's message.

2. Correct spelling mistakes and dialect silently.

3. Keep important nouns, names, products, services, categories,
   organizations, people, locations, models, codes, and other entities.

4. Do NOT invent facts.

5. Do NOT add a specific entity that the user did not mention.

6. Make the query retrieval-friendly.

7. When the user uses a broad concept, preserve the important concept
   and, when obvious from the wording, include natural equivalent terms.

   Example:
   "مين الدكاتره عندكم؟"
   can become:
   "الأطباء الدكاترة المتاحون"

   Example:
   "ايش المحركات البحرية عندكم؟"
   can become:
   "المحركات البحرية المتوفرة"

8. Do NOT add a business name unless the user explicitly provided it.

9. Do NOT answer the user's question.

10. Return only one short search query.

11. Do not turn an ambiguous question into a more specific question.

12. Preserve product names and proper nouns exactly when possible.

==================================================
TYPO / DIALECT EXAMPLES
==================================================

User:
وين موقكم

Result:
SEARCH_QUERY: وين موقعكم؟

User:
رقمكن كم

Result:
SEARCH_QUERY: كم رقم التواصل؟

User:
ابغا احجز دكتر

Result:
SEARCH_QUERY: أريد حجز موعد مع طبيب

User:
كم سعره

Result:
SEARCH_QUERY: كم سعره؟

User:
ايش الاشياء المتوفره

Result:
SEARCH_QUERY: ما الأشياء المتوفرة؟

User:
مين الدكاتره عندكم؟

Result:
SEARCH_QUERY: الأطباء الدكاترة المتاحون

User:
ايش المنتجات الي عندكم؟

Result:
SEARCH_QUERY: المنتجات المتوفرة

User:
عندكم قطع غيار للمحرك؟

Result:
SEARCH_QUERY: قطع غيار المحرك المتوفرة

==================================================
IMPORTANT
==================================================

If the user asks about a specific domain, preserve that domain from the
user's own words.

Do not assume what the business sells.

Do not assume the customer is asking about healthcare.

Do not assume the customer is asking about products.

Do not assume the customer is asking about people.

Do not assume the customer is asking about a location.

User message:

{cleaned}
"""

    result = _call_claude(prompt, max_tokens=180)

    if not result:
        return cleaned or question

    result = result.strip()

    if result.upper().startswith("CASUAL_CONVERSATION"):
        return CASUAL_MARKER

    if result.upper().startswith("SEARCH_QUERY:"):
        result = result.split(":", 1)[1].strip()

    return result or cleaned or question


# ----------------------------------------------------------------------
# 2) Generic casual conversation
# ----------------------------------------------------------------------

def generate_casual_answer(question: str):
    """
    Handles simple social messages without retrieval.
    """

    prompt = f"""
You are a friendly customer-support assistant.

The user's message is casual conversation only.

Respond naturally and briefly in 1-2 short sentences.

Rules:
- Use the same language as the user.
- Match the user's tone.
- Be warm and natural.
- Do not invent business information.
- Do not provide domain-specific information.
- Do not mention RAG, context, embeddings, knowledge base, or retrieval.
- Do not turn casual conversation into a business answer.

Examples:

User:
كيفك؟

Response:
بخير الحمد لله 🌷 كيف أقدر أساعدك؟

User:
كيف اخبارك؟

Response:
بخير ولله الحمد 🌷 وأنت كيفك؟

User:
يعطيكم العافية

Response:
الله يعافيك 🌷 حياك الله.

User:
شكراً

Response:
العفو، حياك الله 🌷

User:
السلام عليكم

Response:
وعليكم السلام ورحمة الله وبركاته 🌷 حياك الله.

User:
hi

Response:
Hello 🌷 How can I help you?

User:
{_light_clean(question)}
"""

    result = _call_claude(prompt, max_tokens=120)

    if result:
        return result

    return "حياك الله 🌷 كيف أقدر أساعدك؟"


# ----------------------------------------------------------------------
# 3) Generic grounded answer
# ----------------------------------------------------------------------

def generate_answer(question: str, context: str):
    """
    Generate an answer using ONLY the retrieved knowledge.

    No hardcoded business/domain rules.
    """

    prompt = f"""
You are a helpful customer-support assistant for a business.

Answer the user's question using ONLY the information provided
in the context.

The business/domain is determined entirely by the context.
Never assume what the business sells or does.

==================================================
STRICT GROUNDING RULES
==================================================

1. Every factual statement must be supported by the context.

2. NEVER invent:
- prices
- products
- services
- people
- names
- availability
- policies
- locations
- dates
- phone numbers
- emails
- URLs
- procedures
- specifications

3. If the context contains only part of the answer:
- provide the confirmed information
- clearly say that the remaining information is not available

4. Do not use external knowledge to fill gaps.

5. Use the user's language.

6. Understand dialect and spelling mistakes naturally.

7. Keep the response concise and direct.

8. Do not mention:
- RAG
- embeddings
- vector search
- retrieval
- context
- knowledge base
- AI system

9. Do not ask follow-up questions unless absolutely necessary for
the answer. Prefer giving the confirmed information available.

10. If a relevant URL appears in the context, preserve it exactly.

11. If the context contains several unrelated entries, ignore them.

12. Do not copy irrelevant sections from the context.

13. Never turn a statement in the knowledge into a question to the user.

==================================================
CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    result = _call_claude(prompt, max_tokens=600)

    if result:
        return result

    return generate_fallback_answer(
        question,
        "RELATED_BUT_UNKNOWN",
    )


# ----------------------------------------------------------------------
# 4) Generic relevance check
# ----------------------------------------------------------------------

def check_context_relevance(
    question: str,
    context: str,
):
    """
    Decide whether the retrieved context contains useful information
    for answering the user's question.

    Completely domain-agnostic.
    """

    prompt = f"""
Determine whether the provided context contains useful information
that can answer the user's question.

The business/domain is unknown to you except for what can be inferred
from the context.

The question may contain:
- spelling mistakes
- dialect
- short wording
- informal wording

Understand the intended meaning.

Return ONLY one:

ANSWERABLE

or

NOT_ANSWERABLE

Return ANSWERABLE when the context contains:
- the direct answer
- a useful partial answer
- a relevant product/service/person/category
- a relevant price
- a relevant location
- a relevant procedure
- a relevant URL
- other information that directly helps answer the question

Return NOT_ANSWERABLE when the context contains no useful information
for the user's question.

Do not assume a connection just because the context and question share
generic words.

==================================================
CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    result = _call_claude(prompt, max_tokens=20)

    if result:
        result = result.strip().upper()

        if result in {"ANSWERABLE", "NOT_ANSWERABLE"}:
            return result

    return "NOT_ANSWERABLE"


# ----------------------------------------------------------------------
# 5) Generic knowledge-gap classification
# ----------------------------------------------------------------------

def classify_knowledge_gap(
    question: str,
    context: str,
):
    """
    Generic classification for unanswered questions.
    """

    prompt = f"""
Classify the user's question using ONLY the question and the context.

Return exactly one:

RELATED_BUT_UNKNOWN

or

OUT_OF_SCOPE

RELATED_BUT_UNKNOWN:
The question appears to be related to the business/domain represented
by the available context, but the context does not contain enough
information to answer it.

OUT_OF_SCOPE:
The question is clearly unrelated to the business/domain represented
by the context.

Important:
- Do not classify something as OUT_OF_SCOPE merely because the exact
  answer is missing.
- Spelling mistakes and dialect should be ignored.
- Judge the user's intended meaning.

==================================================
CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    result = _call_claude(prompt, max_tokens=30)

    if result:
        result = result.strip().upper()

        if result in {
            "RELATED_BUT_UNKNOWN",
            "OUT_OF_SCOPE",
        }:
            return result

    return "RELATED_BUT_UNKNOWN"


# ----------------------------------------------------------------------
# 6) Generic deterministic fallback
# ----------------------------------------------------------------------

def generate_fallback_answer(
    question: str,
    gap_type: str = "RELATED_BUT_UNKNOWN",
) -> str:
    """
    Safe fallback without any business-specific assumptions.
    """

    arabic = _is_arabic(question)

    if gap_type == "OUT_OF_SCOPE":
        if arabic:
            return (
                "أقدر أساعدك في المعلومات المتعلقة بالخدمة أو الجهة "
                "التي تتحدث معها 🌷"
            )

        return (
            "I can help with information related to the business or "
            "service you are contacting 🌷"
        )

    if arabic:
        return (
            "ما لقيت معلومة مؤكدة عن هذا السؤال حاليًا 🌷"
        )

    return (
        "I couldn't find confirmed information about this right now 🌷"
    )