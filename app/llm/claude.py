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
    """
    Normalize client configuration.

    The RAG/LLM layer must never depend on a specific tenant.
    Every tenant-specific behavior comes through this object.
    """
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
    """
    Convert JSON configuration arrays/strings into prompt-friendly text.
    """

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
    """
    Build the tenant-specific profile injected into Claude.

    This is configuration, not hardcoded business logic.
    """

    name = _get_client_name(client_config)

    description = (
        _get_company_description(client_config)
        or "No company description provided."
    )

    business_rules = _format_rules(
        client_config.get("business_rules"),
    )

    booking_rules = _format_rules(
        client_config.get("booking_rules"),
    )

    safety_rules = _format_rules(
        client_config.get("safety_rules"),
    )

    response_rules = _format_rules(
        client_config.get("response_rules"),
    )

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
    """
    Extract the first text block from Claude.

    Thinking/tool blocks are ignored.
    The application only receives the final textual answer.
    """

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
    """
    Central Claude API wrapper.

    The wrapper never raises into the RAG pipeline.
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
            print(
                f"[claude_service] API error: {e}"
            )
            return None

        text = _extract_text(response)

        if text:
            return text

        print(
            "[claude_service] no text block "
            f"(stop_reason="
            f"{getattr(response, 'stop_reason', None)}), "
            "retrying"
        )

    return None


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------

def _is_arabic(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0600-\u06FF]",
            text or "",
        )
    )


def _light_clean(text: str) -> str:
    """
    Lightweight normalization before sending text to Claude.
    """

    text = (text or "").strip()

    # Remove Arabic diacritics and tatweel.
    text = re.sub(
        r"[\u064B-\u0652\u0640]",
        "",
        text,
    )

    # هلااااا -> هلاا
    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ----------------------------------------------------------------------
# 1) Query understanding / rewriting
# ----------------------------------------------------------------------

def rewrite_query(
    question: str,
    client_config: dict | None = None,
) -> str:
    """
    Understand and rewrite the user's message for retrieval.

    Responsibilities:
    - typo correction
    - dialect normalization
    - intent preservation
    - casual conversation detection
    - retrieval-friendly phrasing

    It must NEVER answer the user's question.
    """

    config = _normalize_client_config(
        client_config
    )

    cleaned = _light_clean(question)

    client_profile = _build_client_profile(
        config
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

==================================================
USER MESSAGE
==================================================

{cleaned}
"""

    result = _call_claude(
        prompt,
        max_tokens=150,
    )

    if not result:
        return cleaned or question

    result = result.strip()

    if result.upper().startswith(
        "CASUAL_CONVERSATION"
    ):
        return CASUAL_MARKER

    if result.startswith(
        "SEARCH_QUERY:"
    ):
        result = result[
            len("SEARCH_QUERY:")
        ].strip()

    return result or cleaned or question


# ----------------------------------------------------------------------
# 2) Casual conversation
# ----------------------------------------------------------------------

def generate_casual_answer(
    question: str,
    client_config: dict | None = None,
):
    """
    Generate a short natural response for purely casual messages.

    No knowledge retrieval is required.
    """

    config = _normalize_client_config(
        client_config
    )

    client_name = _get_client_name(
        config
    )

    description = _get_company_description(
        config
    )

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

    result = _call_claude(
        prompt,
        max_tokens=120,
    )

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
):
    """
    Generate the final answer from retrieved knowledge.

    The model performs internal evidence extraction and verification
    before producing the final answer.

    IMPORTANT:
    The internal reasoning is NOT returned to the user.
    """

    config = _normalize_client_config(
        client_config
    )

    client_profile = _build_client_profile(
        config
    )

    website_url = _get_website_url(
        config
    )

    website_instruction = (
        website_url
        if website_url
        else "No official website is configured."
    )

    prompt = f"""
You are the primary answer-generation component of a
multi-tenant RAG customer-support system.

You are serving this business:

{client_profile}

==================================================
YOUR OBJECTIVE
==================================================

Answer the user's question accurately using the retrieved
knowledge and the configured business rules.

The knowledge context is your primary source for
dynamic factual information.

The client configuration is your source for:

- business identity
- stable business rules
- workflow rules
- safety rules
- response rules

General world knowledge is NOT a source of business facts.

Never invent a business fact because you know it from general
knowledge.

==================================================
INTERNAL REASONING PROCESS
==================================================

Before writing the final answer, reason internally through these
steps:

1. Understand the user's actual intent.

2. Identify the entities, services, products, people,
   prices, locations, dates, URLs, policies, or other
   details the user is asking about.

3. Extract only the context entries that directly help answer
   the question.

4. Distinguish:
   - exact facts
   - relevant partial facts
   - unrelated information
   - examples
   - duplicated entries

5. Prefer specific knowledge entries over generic examples
   when the specific information directly answers the question.

6. If the user asks for a list of entities such as products,
   doctors, branches, services, or available options:
   - inspect ALL relevant retrieved entries
   - deduplicate them
   - include the relevant distinct entries available in context
   - do not answer using only the first matching entry

7. If multiple retrieved entries contain the same information,
   treat them as duplicates, not separate facts.

8. If entries conflict:
   - do not invent a resolution
   - do not average values
   - do not silently combine conflicting facts
   - use explicit source/currentness information if available
   - otherwise communicate the uncertainty briefly

9. Apply the client's configured business, workflow,
   and safety rules.

10. Perform a final hallucination check:
    Every business-specific factual statement in the final answer
    must be supported by either:
    - retrieved context
    - client configuration

Do NOT expose this reasoning process.

Return ONLY the final user-facing answer.

==================================================
STRICT GROUNDING
==================================================

- Never invent prices.
- Never invent names.
- Never invent availability.
- Never invent appointment times.
- Never invent contact information.
- Never invent URLs.
- Never invent product specifications.
- Never invent policies.
- Never invent locations.
- Never invent capabilities.
- Never invent procedures.

If a required detail is not available,
say that it is not confirmed.

If useful partial information exists,
provide the confirmed part instead of refusing completely.

==================================================
URL HANDLING
==================================================

When URLs exist in the knowledge context:

- Copy them exactly.
- Do not modify them.
- Prefer a relevant direct URL over a general website URL.
- Do not create a URL from memory.

The configured official website is:

{website_instruction}

Use it when:
- the user explicitly asks for the website
- the answer is incomplete and the business configuration
  says the website is the appropriate fallback
- the context does not contain a better relevant URL

==================================================
LANGUAGE
==================================================

- Answer in the same language as the user.
- Preserve natural dialect when appropriate.
- Arabic users should receive natural Arabic.
- English users should receive natural English.
- Do not mention spelling mistakes.

==================================================
SAFETY
==================================================

Apply ALL safety rules configured for this client.

Never bypass a configured safety rule simply because
the user explicitly asks for the restricted information.

If the user requests professional advice that the client's
safety rules prohibit:

- do not provide the prohibited advice
- explain briefly
- direct the user toward an appropriate configured service
  or professional workflow when the knowledge/config provides one

Do not invent a replacement service.

==================================================
RESPONSE QUALITY
==================================================

- Answer the actual question directly.
- Do not repeat the user's question unnecessarily.
- Do not ask unnecessary follow-up questions.
- Keep the response concise but complete.
- Use bullets when presenting multiple entities or options.
- Include relevant prices when confirmed.
- Include relevant URLs when confirmed.
- Do not dump unrelated context.
- Do not mention the knowledge base.
- Do not mention RAG.
- Do not mention embeddings.
- Do not mention retrieval.
- Do not mention Claude or the AI system.
- Do not reveal internal reasoning.

==================================================
RETRIEVED KNOWLEDGE
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}
"""

    result = _call_claude(
        prompt,
        max_tokens=700,
    )

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
):
    """
    Determine whether the system has enough reliable information
    to provide a useful answer.

    Sources of truth:
    1. Retrieved knowledge
    2. Client configuration

    The function does NOT generate an answer.
    """

    config = _normalize_client_config(
        client_config
    )

    client_profile = _build_client_profile(
        config
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

    result = _call_claude(
        prompt,
        max_tokens=100,
    )

    if result:
        result = result.strip().upper()

        # Be tolerant if Claude adds whitespace/newlines.
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
    """
    Classify why the current question cannot be answered reliably.
    """

    config = _normalize_client_config(
        client_config
    )

    client_profile = _build_client_profile(
        config
    )

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

    result = _call_claude(
        prompt,
        max_tokens=40,
    )

    if result:
        result = result.strip().upper()

        if result in {
            "RELATED_BUT_UNKNOWN",
            "MEDICAL_ADVICE",
            "OUT_OF_SCOPE",
        }:
            return result

    # Safe default:
    # treat uncertainty as a knowledge gap rather than pretending
    # the user is out of scope.
    return "RELATED_BUT_UNKNOWN"


# ----------------------------------------------------------------------
# 6) Deterministic fallback
# ----------------------------------------------------------------------

def generate_fallback_answer(
    question: str,
    gap_type: str = "RELATED_BUT_UNKNOWN",
    client_config: dict | None = None,
) -> str:
    """
    Deterministic fallback.

    No LLM is used here, which means the fallback itself
    cannot hallucinate business facts.
    """

    config = _normalize_client_config(
        client_config
    )

    client_name = _get_client_name(
        config
    )

    website_url = _get_website_url(
        config
    )

    description = _get_company_description(
        config
    )

    arabic = _is_arabic(question)

    website_line_ar = (
        f"\n{website_url}"
        if website_url
        else ""
    )

    website_line_en = (
        f"\n{website_url}"
        if website_url
        else ""
    )

    # --------------------------------------------------------------
    # Medical advice
    # --------------------------------------------------------------

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
                    f"\nيمكنك الرجوع إلى الموقع الرسمي:\n"
                    f"{website_url}"
                )

            return response

        response = (
            "Sorry, I can't provide personal medical advice "
            "or a diagnosis 🌷\n"
            "Please use the appropriate professional service "
            "available from the business."
        )

        if website_url:
            response += (
                f"\nOfficial website:\n{website_url}"
            )

        return response

    # --------------------------------------------------------------
    # Out of scope
    # --------------------------------------------------------------

    if gap_type == "OUT_OF_SCOPE":

        if arabic:
            response = (
                f"أنا مساعد {client_name}، وأقدر أساعدك "
                "في الأسئلة المتعلقة بخدمات الجهة ومعلوماتها."
            )

            if description:
                response += (
                    f"\n\n{description}"
                )

            if website_url:
                response += (
                    f"\n\nللمزيد من المعلومات:\n"
                    f"{website_url}"
                )

            return response

        response = (
            f"I'm the {client_name} and I can help with "
            "questions related to the business and its services."
        )

        if description:
            response += (
                f"\n\n{description}"
            )

        if website_url:
            response += (
                f"\n\nFor more information:\n"
                f"{website_url}"
            )

        return response

    # --------------------------------------------------------------
    # Related but unknown
    # --------------------------------------------------------------

    if arabic:
        response = (
            "ما لقيت معلومة مؤكدة عن هذا السؤال حاليًا 🌷"
        )

        if website_url:
            response += (
                "\nتقدر تتأكد من الموقع الرسمي للحصول "
                "على أدق المعلومات:"
                f"\n{website_url}"
            )

        return response

    response = (
        "I couldn't find confirmed information about "
        "this right now 🌷"
    )

    if website_url:
        response += (
            "\nPlease check the official website for "
            "the most accurate information:"
            f"\n{website_url}"
        )

    return response