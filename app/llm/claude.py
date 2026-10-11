import os
import re

import anthropic
from dotenv import load_dotenv


load_dotenv()


client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001",
)

CASUAL_MARKER = "__CASUAL_CONVERSATION__"


# ----------------------------------------------------------------------
# Client configuration helpers
# ----------------------------------------------------------------------

def _normalize_client_config(client_config: dict | None) -> dict:
    return client_config or {}


def _get_client_name(client_config: dict) -> str:
    return (
        client_config.get("assistant_name")
        or client_config.get("name")
        or "Customer Support Assistant"
    )


def _get_company_description(client_config: dict) -> str:
    return (
        client_config.get("company_description")
        or ""
    ).strip()


def _get_website_url(client_config: dict) -> str | None:
    url = (
        client_config.get("website_url")
        or ""
    ).strip()

    return url or None


def _format_rules(
    rules,
    empty_message: str = "No additional rules provided.",
) -> str:
    if not rules:
        return empty_message

    if isinstance(rules, str):
        return f"- {rules}"

    if isinstance(rules, dict):
        return "\n".join(
            f"- {key}: {value}"
            for key, value in rules.items()
        )

    if isinstance(rules, list):
        formatted = []

        for rule in rules:
            if rule is None:
                continue

            if isinstance(rule, dict):
                formatted.append(
                    "- "
                    + ", ".join(
                        f"{key}: {value}"
                        for key, value in rule.items()
                    )
                )
            else:
                formatted.append(f"- {rule}")

        return "\n".join(formatted) or empty_message

    return empty_message


def _build_client_profile(client_config: dict) -> str:
    name = _get_client_name(client_config)

    description = (
        _get_company_description(client_config)
        or "No company description provided."
    )

    business_rules = _format_rules(client_config.get("business_rules"))
    booking_rules = _format_rules(client_config.get("booking_rules"))
    safety_rules = _format_rules(client_config.get("safety_rules"))
    response_rules = _format_rules(client_config.get("response_rules"))

    website_url = _get_website_url(client_config)

    website_section = (
        website_url
        if website_url
        else "No official website configured."
    )

    return f"""
CLIENT / BUSINESS PROFILE
-------------------------
Name:
{name}

Description:
{description}

Official website:
{website_section}

Business rules:
{business_rules}

Booking / workflow rules:
{booking_rules}

Safety rules:
{safety_rules}

Response rules:
{response_rules}
""".strip()


# ----------------------------------------------------------------------
# Claude helpers
# ----------------------------------------------------------------------

def _extract_text(response):
    for block in response.content:
        if (
            getattr(block, "type", None) == "text"
            and getattr(block, "text", None)
        ):
            return block.text.strip()

    return None


def _call_claude(
    prompt: str,
    max_tokens: int,
):
    """Central Claude API wrapper. Never raises into the RAG pipeline."""

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
            f"(stop_reason={getattr(response, 'stop_reason', None)}), "
            "retrying"
        )

    return None


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------

def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _light_clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _history_block(history: str, title: str, note: str) -> str:
    history = (history or "").strip()[-4000:]

    if not history:
        return ""

    return f"""
==================================================
{title}
==================================================

{note}

{history}
"""


# ----------------------------------------------------------------------
# 1) Query understanding / rewriting
# ----------------------------------------------------------------------

def rewrite_query(
    question: str,
    client_config: dict | None = None,
    history: str = "",
) -> str:
    """
    Rewrite the user's message into a retrieval query.
    Uses conversation history only to resolve references.
    It must NEVER answer the user's question.
    """

    config = _normalize_client_config(client_config)
    cleaned = _light_clean(question)
    client_profile = _build_client_profile(config)

    history_block = _history_block(
        history,
        "CONVERSATION HISTORY",
        """Use the history ONLY to resolve references in the user message
(pronouns, omitted subjects, "how much", "is he a specialist",
"هو", "كم سعره", "كم سعرها").

- Replace the reference with the exact name/service/entity from
  the history. This is the one case where you SHOULD make the
  query more specific.
- If the message answers a question the assistant asked, combine
  both into one full query.
- A reply to the assistant's question (yes / no / a name / a city)
  is NOT casual conversation.
- Copy names and terms exactly as written in the history.
- Do not invent anything that is not in the history.
- If the history does not identify the reference, keep the message
  as it is. Do not guess.""",
    )

    prompt = f"""
You are the query-understanding component of a
multi-tenant business RAG system.

Your job is NOT to answer the user.

Your job is to understand what the user means and produce
the best short retrieval query for the knowledge base.

You must work for completely different businesses and domains.

{client_profile}

==================================================
CORE TASK
==================================================

Determine whether the user message is:

1. A REAL INFORMATION / BUSINESS / SERVICE QUERY
2. A CASUAL CONVERSATIONAL MESSAGE

Return ONLY ONE of:

SEARCH_QUERY: <rewritten query>

or

CASUAL_CONVERSATION

==================================================
TYPO / DIALECT UNDERSTANDING
==================================================

Users may write:

- spelling mistakes
- missing letters
- extra letters
- swapped letters
- wrong Arabic endings
- merged words
- dialect
- Arabizi
- English words written phonetically in Arabic
- informal abbreviations

You must infer the intended meaning.

Example:

User:
وين موقكم

Good retrieval query:
SEARCH_QUERY: ما هو موقع الشركة؟

User:
رقمكن كم

Good retrieval query:
SEARCH_QUERY: ما هو رقم التواصل؟

User:
ابغا احجز

Good retrieval query:
SEARCH_QUERY: أرغب في الحجز.

User:
كم السعر

Good retrieval query:
SEARCH_QUERY: كم السعر؟

Do not preserve obvious spelling mistakes.

==================================================
SEMANTIC PRESERVATION
==================================================

Preserve the user's actual intent.

Do NOT:

- invent facts
- invent services
- invent products
- invent people
- invent locations
- invent policies
- invent prices
- invent entities
- answer the question
- make the question more specific than the user did
  (except to resolve a reference using CONVERSATION HISTORY)

If the user's wording is ambiguous, preserve the ambiguity.

Example:

User:
كيف اعرف مواعيد الادويه

Return:
SEARCH_QUERY: كيف أعرف مواعيد الأدوية؟

Do NOT decide whether this means:
- medication schedule
- medication delivery
- prescription schedule

unless the user explicitly clarified it.

==================================================
CASUAL CONVERSATION
==================================================

Use CASUAL_CONVERSATION only when the user is
simply social and does not need business information.

Examples:

كيفك؟
كيف حالك؟
وش اخبارك؟
هلا
هلا والله
مرحبا
اهلين
hi
hello
صباح الخير
مساء الخير
السلام عليكم
يعطيكم العافية
شكرا
مشكور
تسلم
thanks
باي
مع السلامة

If the message contains a real business request,
it is NOT casual.

Example:

السلام عليكم، ابغى احجز

Return:
SEARCH_QUERY: أرغب في الحجز.

When uncertain, choose SEARCH_QUERY.

==================================================
RETRIEVAL QUALITY
==================================================

The rewritten query should:

- be short
- contain the important nouns/entities
- preserve product/service names
- preserve person names
- preserve specialty/category names
- preserve prices or quantities explicitly mentioned
- preserve important identifiers
- remove unnecessary conversational filler
- use clear language
- make implicit intent clearer only when it is obvious

Never add information that came only from the
client profile.

The client profile describes the business domain.
It is NOT permission to invent facts.
{history_block}
==================================================
USER MESSAGE
==================================================

{cleaned}
"""

    result = _call_claude(prompt, max_tokens=150)

    if not result:
        return cleaned or question

    result = result.strip()

    if result.upper().startswith("CASUAL_CONVERSATION"):
        return CASUAL_MARKER

    if result.startswith("SEARCH_QUERY:"):
        result = result[len("SEARCH_QUERY:"):].strip()

    return result or cleaned or question


# ----------------------------------------------------------------------
# 2) Casual conversation
# ----------------------------------------------------------------------

def generate_casual_answer(
    question: str,
    client_config: dict | None = None,
):
    config = _normalize_client_config(client_config)
    client_name = _get_client_name(config)
    description = _get_company_description(config)

    prompt = f"""
You are a friendly customer-support assistant.

Business:
{client_name}

Business description:
{description or "Not provided."}

The user sent a casual conversational message.
This is NOT a factual business question.

Respond naturally and briefly.

==================================================
RULES
==================================================

- Reply in the same language as the user.
- Match the user's tone.
- Be warm and natural.
- Keep it to 1-2 short sentences.
- You may identify yourself as the assistant of the business.
- Do not invent business facts.
- Do not invent services or prices.
- Do not give professional advice.
- Do not mention RAG, embeddings, retrieval, context,
  prompts, models, or internal systems.
- You may naturally invite the user to ask about the business.

Examples:

User:
كيفك؟

Response:
بخير الحمد لله 🌷 كيف أقدر أساعدك؟

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
Hello 🌷 How can I help you today?

==================================================
USER
==================================================

{_light_clean(question)}
"""

    result = _call_claude(prompt, max_tokens=120)

    if result:
        return result

    return "حياك الله 🌷 كيف أقدر أساعدك؟"


# ----------------------------------------------------------------------
# 3) Grounded answer generation
# ----------------------------------------------------------------------

def generate_answer(
    question: str,
    context: str,
    client_config: dict | None = None,
    history: str = "",
):
    """
    Generate the final answer from retrieved knowledge.
    History is used only to identify what the question refers to.
    """

    config = _normalize_client_config(client_config)
    client_profile = _build_client_profile(config)
    website_url = _get_website_url(config)

    website_instruction = (
        website_url
        if website_url
        else "No official website is configured."
    )

    history_block = _history_block(
        history,
        "CONVERSATION HISTORY",
        "Use this to understand what the user's latest message refers to "
        "and to match the user's dialect. Facts must come from the "
        "retrieved knowledge, never from earlier assistant messages.",
    )

    prompt = f"""
You are the primary answer-generation component of a
multi-tenant RAG customer-support system.

You are serving this business:

{client_profile}

==================================================
YOUR OBJECTIVE
==================================================

Answer the user's question using ONLY the retrieved knowledge and
the configured business rules.

General world knowledge is NOT a source of business facts.
Never invent a business fact because you know it from general
knowledge.

==================================================
DECIDE FIRST
==================================================

Look at the latest user message and the conversation history (if any):

1. Clear question, and the knowledge answers it
   -> answer from the knowledge.

2. Unclear alone, but the history identifies what it refers to
   -> use that subject and answer from the knowledge.

3. Unclear (no subject, or too vague to answer) and the history does
   not identify it
   -> do not guess and do not list unrelated items. Say briefly what
   the answer depends on (test name, city, service, doctor), then ask
   the user to clarify in one short sentence.

4. The user asks about a SPECIFIC item (a test, doctor, service,
   product, insurance company, medicine, branch, offer) and that item
   does NOT appear in the knowledge
   -> treat the knowledge as the complete list of what the business
   has. Say clearly that it is not available with us (for example:
   "تحليل X غير متوفر لدينا حالياً"). If the knowledge has related
   available items of the same kind, mention them briefly as
   alternatives. Do not invent the item, its price, or availability.

5. A business-related question the knowledge does not cover at all
   (and it is not a specific missing item as in 4)
   -> do not invent an answer. Refer the user to the official website:
   {website_instruction}
   in one short sentence.

6. Unrelated to the business
   -> say politely that you can only help with the business services
   and information.

Say "I don't know" or "not confirmed" ONLY when the knowledge truly
has nothing on the subject. Never say you did not understand a clear
question. If the knowledge has a partial answer, give the confirmed
part.

==================================================
STRICT GROUNDING
==================================================

Every business-specific statement must come from the retrieved
knowledge or the client configuration.

- Never invent prices, names, availability, appointment times,
  contact information, URLs, specifications, policies, locations,
  capabilities, or procedures.
- Copy URLs exactly as they appear in the knowledge. Never build a
  URL from memory. The only other allowed URL is the configured
  official website.
- If entries conflict, do not average or merge them; state the
  uncertainty briefly.
- Duplicate entries are one fact, not several.
- If the user asks for a list (doctors, services, options), include
  all the distinct relevant entries from the knowledge, not only
  the first.
- Prefer specific entries over generic ones.

==================================================
CONTEXT AND DIALECT
==================================================

- Always read the latest message in the context of the conversation.
- Answer in the same language as the user.
- Match the user's dialect and tone (Saudi, Hijazi, Gulf, Egyptian,
  etc.). If the user writes casually, answer casually. Keep it
  natural and polite, not stiff.
- Do not mention spelling mistakes.

==================================================
SAFETY
==================================================

Apply ALL safety rules configured for this client.

Never bypass a configured safety rule simply because the user asks
for the restricted information. If the user requests professional
advice that the safety rules prohibit: do not provide it, explain
briefly, and direct the user to the appropriate service or workflow
that the knowledge/config provides. Do not invent a replacement
service.

==================================================
RESPONSE QUALITY
==================================================

- Answer the actual question directly and concisely.
- Do not repeat the user's question.
- Use bullets only for multiple items or options.
- Include relevant prices and URLs when they are in the knowledge.
- Plain text only: no Markdown bold, no asterisks.
- Do not dump unrelated information.
- Do not mention the knowledge base, RAG, embeddings, retrieval,
  Claude, or the AI system. Do not reveal internal reasoning.
- Return ONLY the final user-facing answer.
{history_block}
==================================================
RETRIEVED KNOWLEDGE
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    result = _call_claude(prompt, max_tokens=700)

    if result:
        return result

    return generate_fallback_answer(
        question=question,
        gap_type="RELATED_BUT_UNKNOWN",
        client_config=config,
    )

    result = _call_claude(prompt, max_tokens=700)

    if result:
        return result

    return generate_fallback_answer(
        question=question,
        gap_type="RELATED_BUT_UNKNOWN",
        client_config=config,
    )


# ----------------------------------------------------------------------
# 4) Context relevance
# ----------------------------------------------------------------------

def check_context_relevance(
    question: str,
    context: str,
    client_config: dict | None = None,
    history: str = "",
):
    """
    Decide whether the knowledge/config can produce a useful answer.
    """

    config = _normalize_client_config(client_config)
    client_profile = _build_client_profile(config)

    history_block = _history_block(
        history,
        "CONVERSATION HISTORY",
        "Use this only to identify what the user's question refers to.",
    )

    prompt = f"""
You are an evidence evaluator in a multi-tenant RAG system.

Your job is to determine whether the system has enough reliable
information to answer the user's question.

You have TWO valid sources of information:

SOURCE A — RETRIEVED KNOWLEDGE
This contains dynamic information retrieved from the client's
knowledge base.

SOURCE B — CLIENT CONFIGURATION
This contains stable information explicitly configured for this
client, such as:
- company identity
- official website
- business description
- stable business rules
- workflow rules
- safety rules
- response rules

Both sources are trusted.

==================================================
DECISION
==================================================

Return ONLY:

ANSWERABLE

or

NOT_ANSWERABLE

==================================================
ANSWERABLE
==================================================

Return ANSWERABLE when either source contains information
that can contribute to a correct user-facing answer.

Examples:

- The context directly answers the question.
- The context contains a useful partial answer.
- The context contains a relevant URL.
- The context contains relevant prices.
- The context contains relevant entities.
- The client configuration contains the official website
  and the user asks for the website.
- The client configuration contains a stable business rule
  directly relevant to the question.
- The combination of client configuration + retrieved
  knowledge allows a useful grounded answer.

The answer does NOT need to contain every possible detail.

==================================================
NOT_ANSWERABLE
==================================================

Return NOT_ANSWERABLE only when neither the retrieved knowledge
nor the client configuration contains useful information for
the question.

Do not reject an answer simply because:

- wording is different
- the user has spelling mistakes
- the user uses dialect
- the answer is distributed across multiple entries
- the knowledge contains extra irrelevant entries

==================================================
UNDER-SPECIFIED QUESTIONS
==================================================

Return ANSWERABLE if the question is clear, OR the history identifies
its subject and the knowledge covers it, OR the question lacks a
subject but the knowledge says which detail the user must provide
(asking for that detail is a valid answer).
Return NOT_ANSWERABLE only when the knowledge has nothing useful.

==================================================
IMPORTANT EVIDENCE RULE
==================================================

Do not judge based on whether the entire context looks relevant.

Look for ANY specific evidence that can answer the question.

For example:

Question:
كيف طرق الدفع

Context:
- مدى
- Visa
- Mastercard
- Apple Pay

Correct result:
ANSWERABLE

Question:
وين موقعكم

Client configuration:
Official website:
https://example.com

Correct result:
ANSWERABLE

Question:
كم سعر المنتج X

Context:
No information about product X.

Client configuration:
No price for product X.

Correct result:
NOT_ANSWERABLE

==================================================
CLIENT CONFIGURATION
==================================================

{client_profile}
{history_block}
==================================================
RETRIEVED KNOWLEDGE
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}

==================================================
OUTPUT
==================================================

Return ONLY:
ANSWERABLE
or
NOT_ANSWERABLE
"""

    result = _call_claude(prompt, max_tokens=100)

    if result:
        result = result.strip().upper()
        first_line = result.splitlines()[0].strip()

        if first_line == "ANSWERABLE":
            return "ANSWERABLE"

        if first_line == "NOT_ANSWERABLE":
            return "NOT_ANSWERABLE"

    return "NOT_ANSWERABLE"


# ----------------------------------------------------------------------
# 5) Knowledge-gap classification
# ----------------------------------------------------------------------

def classify_knowledge_gap(
    question: str,
    context: str,
    client_config: dict | None = None,
):
    config = _normalize_client_config(client_config)
    client_profile = _build_client_profile(config)

    prompt = f"""
You are a knowledge-gap classifier for a multi-tenant
customer-support RAG system.

CLIENT PROFILE:
{client_profile}

Classify the user's question into exactly ONE category.

Return ONLY:

RELATED_BUT_UNKNOWN
MEDICAL_ADVICE
OUT_OF_SCOPE

==================================================
RELATED_BUT_UNKNOWN
==================================================

Use this when:

- the question is about this business/domain
- but the retrieved knowledge does not contain enough
  reliable information to answer it

Examples:

- asking for a business service that is not documented
- asking for a price that is not available
- asking for a person/entity that was not retrieved
- asking for availability that was not retrieved
- asking for a policy not present in knowledge

==================================================
MEDICAL_ADVICE
==================================================

Use this ONLY when the user is asking for personal medical
judgment such as:

- diagnosis
- medication recommendation
- medication dosage
- treatment recommendation
- interpretation requiring personal medical judgment
- asking whether they can take/use a specific medication

This category is not dependent on any particular client.

==================================================
OUT_OF_SCOPE
==================================================

Use this when the question is clearly unrelated to the
business/domain represented by the client profile.

Examples:

- unrelated trivia
- unrelated entertainment
- unrelated sports
- unrelated cooking
- unrelated general topics

Do NOT classify a business question as OUT_OF_SCOPE
just because the answer is missing.

==================================================
CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    result = _call_claude(prompt, max_tokens=40)

    if result:
        result = result.strip().upper()

        if result in {
            "RELATED_BUT_UNKNOWN",
            "MEDICAL_ADVICE",
            "OUT_OF_SCOPE",
        }:
            return result

    return "RELATED_BUT_UNKNOWN"


# ----------------------------------------------------------------------
# 6) Deterministic fallback
# ----------------------------------------------------------------------

def generate_fallback_answer(
    question: str,
    gap_type: str = "RELATED_BUT_UNKNOWN",
    client_config: dict | None = None,
) -> str:
    """Deterministic fallback. No LLM, so it cannot hallucinate."""

    config = _normalize_client_config(client_config)
    client_name = _get_client_name(config)
    website_url = _get_website_url(config)
    description = _get_company_description(config)
    arabic = _is_arabic(question)

    if gap_type == "MEDICAL_ADVICE":

        if arabic:
            response = (
                "عذرًا، ما أقدر أقدّم نصيحة طبية شخصية أو تشخيصًا 🌷\n"
                "للحصول على المساعدة المناسبة، يُفضّل التواصل "
                "مع مختص أو استخدام الخدمة المناسبة المتوفرة "
                "لدى الجهة."
            )

            if website_url:
                response += (
                    f"\nيمكنك الرجوع إلى الموقع الرسمي:\n{website_url}"
                )

            return response

        response = (
            "Sorry, I can't provide personal medical advice "
            "or a diagnosis 🌷\n"
            "Please use the appropriate professional service "
            "available from the business."
        )

        if website_url:
            response += f"\nOfficial website:\n{website_url}"

        return response

    if gap_type == "OUT_OF_SCOPE":

        if arabic:
            response = (
                f"أنا مساعد {client_name}، وأقدر أساعدك "
                "في الأسئلة المتعلقة بخدمات الجهة ومعلوماتها."
            )

            if description:
                response += f"\n\n{description}"

            if website_url:
                response += f"\n\nللمزيد من المعلومات:\n{website_url}"

            return response

        response = (
            f"I'm the {client_name} and I can help with "
            "questions related to the business and its services."
        )

        if description:
            response += f"\n\n{description}"

        if website_url:
            response += f"\n\nFor more information:\n{website_url}"

        return response

    # RELATED_BUT_UNKNOWN: ask for clarification and point to the website.
    if arabic:
        response = "ممكن توضّح سؤالك أكثر عشان أساعدك؟ 🌷"

        if website_url:
            response += (
                "\nوتقدر تتصفح الموقع الرسمي:\n"
                f"{website_url}"
            )

        return response

    response = "Could you clarify your question a bit so I can help? 🌷"

    if website_url:
        response += f"\nYou can also check the official website:\n{website_url}"

    return response
    """Deterministic fallback. No LLM, so it cannot hallucinate."""

    config = _normalize_client_config(client_config)
    client_name = _get_client_name(config)
    website_url = _get_website_url(config)
    description = _get_company_description(config)
    arabic = _is_arabic(question)

    if gap_type == "MEDICAL_ADVICE":

        if arabic:
            response = (
                "عذرًا، ما أقدر أقدّم نصيحة طبية شخصية أو تشخيصًا 🌷\n"
                "للحصول على المساعدة المناسبة، يُفضّل التواصل "
                "مع مختص أو استخدام الخدمة المناسبة المتوفرة "
                "لدى الجهة."
            )

            if website_url:
                response += (
                    f"\nيمكنك الرجوع إلى الموقع الرسمي:\n{website_url}"
                )

            return response

        response = (
            "Sorry, I can't provide personal medical advice "
            "or a diagnosis 🌷\n"
            "Please use the appropriate professional service "
            "available from the business."
        )

        if website_url:
            response += f"\nOfficial website:\n{website_url}"

        return response

    if gap_type == "OUT_OF_SCOPE":

        if arabic:
            response = (
                f"أنا مساعد {client_name}، وأقدر أساعدك "
                "في الأسئلة المتعلقة بخدمات الجهة ومعلوماتها."
            )

            if description:
                response += f"\n\n{description}"

            if website_url:
                response += f"\n\nللمزيد من المعلومات:\n{website_url}"

            return response

        response = (
            f"I'm the {client_name} and I can help with "
            "questions related to the business and its services."
        )

        if description:
            response += f"\n\n{description}"

        if website_url:
            response += f"\n\nFor more information:\n{website_url}"

        return response

    if arabic:
        response = "ما لقيت معلومة مؤكدة عن هذا السؤال حاليًا 🌷"

        if website_url:
            response += (
                "\nتقدر تتأكد من الموقع الرسمي للحصول "
                f"على أدق المعلومات:\n{website_url}"
            )

        return response

    response = "I couldn't find confirmed information about this right now 🌷"

    if website_url:
        response += (
            "\nPlease check the official website for "
            f"the most accurate information:\n{website_url}"
        )

    return response