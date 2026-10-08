import os

import anthropic
from dotenv import load_dotenv


load_dotenv()


client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def _extract_text(response):
    """
    Extract the first text block from a Claude response.
    Ignores thinking/tool blocks.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            return block.text.strip()

    return None


def rewrite_query(question: str) -> str:
    """
    Rewrite a user question for better retrieval.

    Returns:
        - A rewritten search query
        - "__CASUAL_CONVERSATION__" for simple social/conversational messages
    """

    prompt = f"""
You are a query understanding and rewriting component for a Hakeem Care
RAG system.

Your job is to understand the user's message and decide whether it is:

1. A REAL INFORMATION / SERVICE QUERY
2. A CASUAL CONVERSATIONAL MESSAGE

Return ONLY ONE of the following:

SEARCH_QUERY: <rewritten query>

or

CASUAL_CONVERSATION

==================================================
CASUAL CONVERSATION
==================================================

Use CASUAL_CONVERSATION only when the user is simply being social,
friendly, polite, or conversational and is NOT asking for information
about Hakeem Care, healthcare, services, doctors, prices, appointments,
labs, prescriptions, payments, locations, or any other actual request.

Examples:

User:
كيفك؟

Result:
CASUAL_CONVERSATION

User:
كيف حالك؟

Result:
CASUAL_CONVERSATION

User:
كيف اخبارك؟

Result:
CASUAL_CONVERSATION

User:
كيفك يا غالي؟

Result:
CASUAL_CONVERSATION

User:
هلا

Result:
CASUAL_CONVERSATION

User:
يا هلا والله

Result:
CASUAL_CONVERSATION

User:
يعطيكم العافية

Result:
CASUAL_CONVERSATION

User:
شكراً

Result:
CASUAL_CONVERSATION

User:
السلام عليكم

Result:
CASUAL_CONVERSATION

IMPORTANT:

Do NOT classify something as casual if it contains an actual request.

For example:

User:
كيف احجز دكتور؟

Result:
SEARCH_QUERY: كيف يمكن حجز موعد مع طبيب؟

User:
كيف أعرف أسعاركم؟

Result:
SEARCH_QUERY: ما هي أسعار الخدمات؟

User:
كيف حال الطبيب؟

Result:
SEARCH_QUERY: كيف هي حالة الطبيب؟

User:
السلام عليكم، ابغى دكتور جلدية

Result:
SEARCH_QUERY: أريد حجز استشارة في تخصص الجلدية.

==================================================
SEARCH QUERY
==================================================

For real questions:

- Preserve the exact meaning and intent.
- Convert Saudi/Gulf colloquial Arabic into clear Arabic.
- Normalize spelling mistakes and dialect variations.
- Do not add facts that were not mentioned.
- Do not invent services, doctors, prices, locations, or entities.
- Keep important names, services, specialties, products, and entities.
- Do not make the query more specific than the original.
- Make short ambiguous questions clearer only when the intent is obvious.
- Return one short search query.
- Do NOT answer the user's question.

Examples:

User:
شو رقمكن؟

Result:
SEARCH_QUERY: ما هو رقم التواصل؟

User:
وين موقعكم؟

Result:
SEARCH_QUERY: ما هو الموقع الإلكتروني لحكيم كير؟

User:
ايش ايميلكم؟

Result:
SEARCH_QUERY: ما هو البريد الإلكتروني لحكيم كير؟

User:
عندكم دكتور مسالك؟

Result:
SEARCH_QUERY: هل تتوفر استشارة في تخصص المسالك البولية؟

User:
ابغى احجز دكتور

Result:
SEARCH_QUERY: أرغب في حجز موعد مع طبيب.

User:
كم سعر قراءة التحاليل؟

Result:
SEARCH_QUERY: كم سعر قراءة نتائج التحاليل؟

User:
عندكم تحليل فيتامين د؟

Result:
SEARCH_QUERY: هل يتوفر تحليل فيتامين د؟

User:
ما وصلني رابط الموعد

Result:
SEARCH_QUERY: لم يصلني رابط موعد الطبيب.

User:
كيف اقدر اسأل دكتور؟

Result:
SEARCH_QUERY: كيف يمكن استخدام خدمة اسأل طبيب؟

User:
وش الخدمات اللي عندكم؟

Result:
SEARCH_QUERY: ما هي الخدمات التي تقدمها حكيم كير؟

User:
هل عندكم زيارة منزلية؟

Result:
SEARCH_QUERY: هل تتوفر خدمة الزيارة المنزلية؟

User:
ابغى اعرف عن المختبر

Result:
SEARCH_QUERY: أرغب في معرفة معلومات عن خدمات المختبر.

User question:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    result = _extract_text(response)

    if not result:
        # Safe fallback:
        # If Claude cannot rewrite the query, retrieve using the original.
        return question

    result = result.strip()

    if result == "CASUAL_CONVERSATION":
        return "__CASUAL_CONVERSATION__"

    if result.startswith("SEARCH_QUERY:"):
        result = result[len("SEARCH_QUERY:"):].strip()

    return result or question


def generate_casual_answer(question: str):
    """
    Answer simple conversational messages directly.
    No knowledge retrieval is required.
    """

    prompt = f"""
You are a friendly and natural customer support assistant for Hakeem Care.

The user's message is casual conversation only.

Respond naturally and briefly.

Rules:
- Use the same language as the user.
- Match the user's tone.
- Be friendly and polite.
- Do not invent Hakeem Care information.
- Do not mention RAG, context, knowledge base, embeddings, or retrieval.
- Do not turn a casual message into a business answer.
- If appropriate, naturally invite the user to ask their question.

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
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    result = _extract_text(response)

    if result:
        return result

    return "حياك الله 🌷 كيف أقدر أساعدك؟"


def generate_answer(question: str, context: str):
    prompt = f"""
You are a helpful customer support assistant for Hakeem Care.

Your job is to answer the user's question using ONLY the information
provided in the context and the business rules below.

IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

For telehealth appointments, patients do NOT need to visit a medical
center. A patient can book a doctor remotely and attend the consultation
online.

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


IMPORTANT HAKEEM CARE SERVICE RULES:


1. URGENT CONSULTATION

An urgent consultation is available through the Hakeem Care
application or website for 15 SAR.


2. SPECIFIC DOCTOR OR SPECIALTY

When the user asks for a doctor in a specific specialty:

- Explain how to book the online consultation.
- Tell the user to open the Hakeem Care application or website.
- Select Online Consultations.
- Select the requested specialty.
- Select the available doctor.
- Select the suitable appointment.
- Complete the booking.

If the context contains a specific doctor, price, or available
appointment times, include those details.

Do NOT invent doctor names, prices, or appointment times.

A specific online consultation does NOT require asking for the
user's city.


3. LAB RESULT INTERPRETATION

Reading or interpreting laboratory results costs 15 SAR.

Do not confuse laboratory result interpretation with the price of
a laboratory test itself.


4. ASK A DOCTOR

"Ask a Doctor" is a free service available through the Hakeem Care
application only.

It is a chat-based consultation with a doctor and does NOT include
a medical prescription.


5. SERVICE PRICES

When the user asks about prices:

- Use the actual prices available in the context.
- If multiple specific tests or packages are available in the context,
  list them clearly.
- Do not invent a general price when only specific test/package prices
  are available.
- If the available information is partial, clearly explain that prices
  vary by test or package and provide the prices that are actually
  available in the context.


IMPORTANT LINK RULES:

- Inspect the entire context for URLs.
- If a URL is relevant to the user's question, include it.
- If the user asks for Hakeem Care's website, provide the website URL
  found in the context.
- If the user asks how to book and a relevant booking URL exists,
  provide it.
- If a direct service URL exists, prefer it over a general website URL.
- Never invent a URL.
- Never modify a URL.
- Copy URLs exactly as they appear in the context.
- Do not omit a relevant URL just because it appears in a different
  knowledge entry.


GENERAL RULES:

- Answer the user's question directly.
- Do NOT ask follow-up questions.
- Do NOT ask clarification questions unless absolutely required
  by the requested action.
- Do NOT ask for the user's city for general telehealth booking.
- Do not invent information.
- Do not make up prices, doctors, locations, policies, or procedures.
- Keep the answer focused and concise.
- If the context contains useful partial information, use it.
- Clearly distinguish between confirmed information and unavailable
  information.
- Use the same language as the user.
- If the user asks in Arabic, answer in Arabic.
- If the user asks in English, answer in English.
- Do not mention the knowledge base, context, embeddings, retrieval,
  RAG, or AI system.

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

    result = _extract_text(response)

    if result:
        return result

    return "عذرًا، لم أتمكن من إعداد إجابة الآن."


def check_context_relevance(
    question: str,
    context: str,
):
    prompt = f"""
Determine whether the provided context contains useful information
that can be used to answer the user's question.

Return ONLY one of these two values:

ANSWERABLE
NOT_ANSWERABLE

IMPORTANT:

ANSWERABLE does NOT require the context to contain every possible
detail.

Return ANSWERABLE when the context contains:

- The direct answer to the question, OR
- Relevant information that allows a useful partial answer, OR
- A relevant URL or website address, OR
- Relevant prices or service information, OR
- Relevant booking instructions, doctor information, or appointment
  information.

Return NOT_ANSWERABLE ONLY when the context contains no useful
information that can answer the user's question.

IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.

Patients can book doctors remotely and attend consultations online.

A patient does NOT need to visit a medical center for a telehealth
consultation.

Do not mark a telehealth booking question as NOT_ANSWERABLE simply
because the user's city or location is unknown.

IMPORTANT URL RULE:

If a relevant URL appears anywhere in the context, consider the
question ANSWERABLE when that URL helps answer the user's request.

IMPORTANT PRICE RULE:

If the context contains specific prices that are relevant to the
question, consider the question ANSWERABLE even if the context does
not contain every possible test or package price.

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

    result = _extract_text(response)

    if result:
        result = result.strip().upper()

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

The user's question is related to Hakeem Care, its services,
products, doctors, consultations, laboratories, prescriptions,
payments, appointments, or healthcare domain, but the available
context does not contain enough useful information to answer it.

OUT_OF_SCOPE means:

The user's question is clearly unrelated to Hakeem Care,
its services, products, or healthcare domain.

Examples:

Context:
The company provides healthcare services.

Question:
What doctors are available today?

Result:
RELATED_BUT_UNKNOWN


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

A missing city or location does NOT make a general telehealth
question OUT_OF_SCOPE.

Questions about booking doctors, consultations, laboratories,
prescriptions, pharmacies, payments, prices, or Hakeem Care services
are normally RELATED_BUT_UNKNOWN when the exact information is missing.

Do not classify a question as OUT_OF_SCOPE simply because the exact
answer is missing.

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

    result = _extract_text(response)

    if result:
        result = result.strip().upper()

        if result in {
            "RELATED_BUT_UNKNOWN",
            "OUT_OF_SCOPE",
        }:
            return result

    return "OUT_OF_SCOPE"