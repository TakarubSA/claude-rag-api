import os
import re

import anthropic
from dotenv import load_dotenv


load_dotenv()


client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

MODEL = "claude-sonnet-5-5"
WEBSITE_URL = "https://hakeemcare.com/"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _extract_text(response):
    """
    Extract the first text block from a Claude response.
    Ignores thinking/tool blocks.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            return block.text.strip()

    return None


def _call_claude(prompt: str, max_tokens: int):
    """
    Single place for all Claude calls.
    Returns text or None (never raises), so the pipeline never crashes.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(response)
    except Exception as e:
        print(f"[claude_service] API error: {e}")
        return None


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _light_clean(text: str) -> str:
    """
    Light cleanup before sending to the LLM:
    - remove diacritics and tatweel
    - collapse letters repeated 3+ times (هلاااا -> هلاا)
    - collapse whitespace
    """
    text = (text or "").strip()
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ----------------------------------------------------------------------
# 1) Query rewriting (typo tolerant)
# ----------------------------------------------------------------------

def rewrite_query(question: str) -> str:
    """
    Rewrite a user question for better retrieval.

    Returns:
        - A rewritten search query
        - "__CASUAL_CONVERSATION__" for simple social messages
    """

    cleaned = _light_clean(question)

    prompt = f"""
You are a query understanding and rewriting component for a Hakeem Care
RAG system. Hakeem Care is a Saudi telehealth company (online doctor
consultations, labs, home visits, prescriptions, Ask a Doctor, etc.).

Decide whether the user's message is:

1. A REAL INFORMATION / SERVICE QUERY
2. A CASUAL CONVERSATIONAL MESSAGE

Return ONLY ONE of the following:

SEARCH_QUERY: <rewritten query>

or

CASUAL_CONVERSATION

==================================================
TYPO & DIALECT TOLERANCE (VERY IMPORTANT)
==================================================

Users write fast on phones. Their messages often contain:
- Missing, extra, swapped, or wrong letters
  (موقكم = موقعكم, رقمكن = رقمكم, ايميلكن = ايميلكم,
   حجز = حجزز, دكتر = دكتور, تحاليل = تحليل, اسعارم = اسعاركم)
- Wrong ending letters (ن instead of م, ه instead of ة, etc.)
- Missing spaces or merged words (كمسعر = كم سعر)
- Gulf / Saudi / Egyptian / Levantine dialect words
- English words written in Arabic letters, or Arabizi

You MUST infer the intended word from context and meaning, even if
1-2 letters are wrong. NEVER fail or return an empty result because
of a spelling mistake. Always fix the typo silently and write the
query in clear, correct Arabic.

Examples with typos:

User: وين موقكم
Result: SEARCH_QUERY: ما هو موقع حكيم كير؟

User: رقمكن كم
Result: SEARCH_QUERY: ما هو رقم التواصل؟

User: ايميلكن
Result: SEARCH_QUERY: ما هو البريد الإلكتروني لحكيم كير؟

User: ابغا احجز دكتر
Result: SEARCH_QUERY: أرغب في حجز موعد مع طبيب.

User: كمسعر تحليل فتامين د
Result: SEARCH_QUERY: كم سعر تحليل فيتامين د؟

User: عندكم دكتور جلديه؟
Result: SEARCH_QUERY: هل تتوفر استشارة في تخصص الجلدية؟

User: شلون اسال دكتر
Result: SEARCH_QUERY: كيف يمكن استخدام خدمة اسأل طبيب؟

==================================================
CASUAL CONVERSATION
==================================================

Use CASUAL_CONVERSATION only when the user is simply being social,
friendly, polite, or chatting, and is NOT asking for any information
or action about Hakeem Care, healthcare, services, doctors, prices,
appointments, labs, prescriptions, payments, or locations.

Examples (including dialect variations and typos):

كيفك؟ / كيف حالك / كيف اخبارك / شخبارك / وش اخبارك / كيفك يا غالي
ايش مسوي / وش تسوي / وش عندك / شو الاخبار / وش علومك
هلا / هلا والله / يا هلا والله / مرحبا / اهلين / hi / hello
صباح الخير / مساء الخير / السلام عليكم / سلام
يعطيكم العافية / الله يعافيك / شكرا / مشكور / تسلم / thanks
الحمدلله / تمام / طيب / اوكي / باي / مع السلامة

Result for all of the above: CASUAL_CONVERSATION

IMPORTANT:
- If the message contains an actual request or question about a
  service, it is NOT casual, even if it starts with a greeting.
- If you are unsure, choose SEARCH_QUERY (never CASUAL_CONVERSATION).

Examples:

User: كيف احجز دكتور؟
Result: SEARCH_QUERY: كيف يمكن حجز موعد مع طبيب؟

User: كيف أعرف أسعاركم؟
Result: SEARCH_QUERY: ما هي أسعار الخدمات؟

User: كيف حال الطبيب؟
Result: SEARCH_QUERY: كيف هي حالة الطبيب؟

User: السلام عليكم، ابغى دكتور جلدية
Result: SEARCH_QUERY: أريد حجز استشارة في تخصص الجلدية.

==================================================
SEARCH QUERY RULES
==================================================

- Preserve the exact meaning and intent.
- Convert colloquial Arabic into clear Arabic and fix spelling.
- Do not add facts that were not mentioned.
- Do not invent services, doctors, prices, locations, or entities.
- Keep important names, services, specialties, products, entities.
- Do not make the query more specific than the original.
- Make short ambiguous questions clearer only when the intent is obvious.
- If the user describes symptoms or asks about medication, keep the
  question as is (do not answer it), e.g.
  User: عندي صداع ايش اخذ
  Result: SEARCH_QUERY: عندي صداع ما العلاج المناسب؟
- Return one short search query.
- Do NOT answer the user's question.

More examples:

User: شو رقمكم
Result: SEARCH_QUERY: ما هو رقم التواصل؟

User: عندكم دكتور مسالك؟
Result: SEARCH_QUERY: هل تتوفر استشارة في تخصص المسالك البولية؟

User: ابغى احجز دكتور
Result: SEARCH_QUERY: أرغب في حجز موعد مع طبيب.

User: كم سعر قراءة التحاليل؟
Result: SEARCH_QUERY: كم سعر قراءة نتائج التحاليل؟

User: ما وصلني رابط الموعد
Result: SEARCH_QUERY: لم يصلني رابط موعد الطبيب.

User: وش الخدمات اللي عندكم؟
Result: SEARCH_QUERY: ما هي الخدمات التي تقدمها حكيم كير؟

User: هل عندكم زيارة منزلية؟
Result: SEARCH_QUERY: هل تتوفر خدمة الزيارة المنزلية؟

User: ابغى اعرف عن المختبر
Result: SEARCH_QUERY: أرغب في معرفة معلومات عن خدمات المختبر.

User question:
{cleaned}
"""

    result = _call_claude(prompt, max_tokens=150)

    if not result:
        # Safe fallback: retrieve using the cleaned original question.
        return cleaned or question

    result = result.strip()

    if result.upper().startswith("CASUAL_CONVERSATION"):
        return "__CASUAL_CONVERSATION__"

    if result.startswith("SEARCH_QUERY:"):
        result = result[len("SEARCH_QUERY:"):].strip()

    return result or cleaned or question


# ----------------------------------------------------------------------
# 2) Casual conversation
# ----------------------------------------------------------------------

def generate_casual_answer(question: str):
    """
    Answer simple conversational messages directly.
    No knowledge retrieval is required.
    """

    prompt = f"""
You are a friendly and natural customer support assistant for Hakeem Care
(a Saudi telehealth company).

The user's message is casual conversation only (greetings, how are you,
what are you doing, thanks, goodbye, etc.). Messages may contain typos
or dialect: understand the intent anyway.

Respond naturally and briefly (1-2 short sentences).

Rules:
- Use the same language and dialect style as the user (Saudi/Gulf
  Arabic if they write Gulf Arabic).
- Match the user's tone. Be warm and polite.
- You may say you are the Hakeem Care assistant and are doing well,
  if asked how you are or what you are doing.
- Do not invent Hakeem Care information, prices, or services.
- Do not give any medical advice.
- Do not mention RAG, context, knowledge base, embeddings, or retrieval.
- Do not turn a casual message into a business answer.
- Naturally invite the user to ask about Hakeem Care services.

Examples:

User: كيفك؟
Response: بخير الحمد لله 🌷 كيف أقدر أساعدك؟

User: ايش مسوي؟
Response: الحمد لله تمام 🌷 جاهز أساعدك بأي شي تحتاجه من خدمات حكيم كير.

User: كيف اخبارك؟
Response: بخير ولله الحمد 🌷 وأنت كيفك؟

User: يعطيكم العافية
Response: الله يعافيك 🌷 حياك الله.

User: شكراً
Response: العفو، حياك الله 🌷

User: السلام عليكم
Response: وعليكم السلام ورحمة الله وبركاته 🌷 حياك الله.

User: hi
Response: Hello 🌷 How can I help you today?

User:
{_light_clean(question)}
"""

    result = _call_claude(prompt, max_tokens=120)

    if result:
        return result

    return "حياك الله 🌷 كيف أقدر أساعدك؟"


# ----------------------------------------------------------------------
# 3) Grounded answer
# ----------------------------------------------------------------------

def generate_answer(question: str, context: str):
    prompt = f"""
You are a helpful customer support assistant for Hakeem Care.

Your job is to answer the user's question using ONLY the information
provided in the context and the business rules below.

The user's message may contain spelling mistakes or dialect (for example
"موقكم" meaning "موقعكم", or "رقمكن" meaning "رقمكم"). Understand the
intended meaning and answer normally. Never mention the typo.

==================================================
STRICT GROUNDING RULES (MOST IMPORTANT)
==================================================

- Every fact in your answer (prices, doctors, phone numbers, emails,
  policies, procedures, links) must come from the context or from the
  business rules below.
- NEVER give medical advice, diagnosis, medication names, doses, or
  treatment suggestions, even if the user asks directly or describes
  symptoms. You are not a doctor. Instead:
    * Politely say you cannot give medical advice.
    * Point the user to the right Hakeem Care service:
      urgent consultation (15 SAR), booking a specific specialty
      online, or "Ask a Doctor" (free, app only).
- NEVER guess or fill gaps from your own knowledge.
- If the context does NOT contain the answer (or only part of it):
    * Share only what is confirmed in the context.
    * Clearly say the rest is not available to you right now.
    * Tell the user to check the Hakeem Care website for the most
      accurate information: {WEBSITE_URL}
- Always include {WEBSITE_URL} when you cannot fully answer.

==================================================
BUSINESS CONTEXT
==================================================

Hakeem Care provides remote telehealth consultations.

For telehealth appointments, patients do NOT need to visit a medical
center. A patient can book a doctor remotely and attend the
consultation online.

Do NOT assume that a patient needs to visit a branch or medical center
when asking about booking a doctor or a medical consultation.

Do NOT ask the patient for their city or location when the request is
about booking a doctor or a remote telehealth consultation.

Physical locations are only relevant when the question is specifically
about a physical service, such as:
- Laboratory branches
- Home visit coverage
- Pharmacy pickup
- Another service that actually requires a physical location

==================================================
SERVICE RULES
==================================================

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
application only. It is a chat-based consultation with a doctor and
does NOT include a medical prescription.

5. SERVICE PRICES
When the user asks about prices:
- Use the actual prices available in the context.
- If multiple specific tests or packages are available, list them
  clearly.
- Do not invent a general price when only specific test/package prices
  are available.
- If the information is partial, explain that prices vary by test or
  package, give the prices that are available, and refer the user to
  {WEBSITE_URL} for the full list.

==================================================
LINK RULES
==================================================

- Inspect the entire context for URLs.
- If a URL is relevant to the user's question, include it.
- If the user asks for the website, provide the website URL found in
  the context (or {WEBSITE_URL}).
- If the user asks how to book and a relevant booking URL exists,
  provide it.
- If a direct service URL exists, prefer it over a general URL.
- Never invent a URL. Never modify a URL.
- Copy URLs exactly as they appear in the context.
- Do not omit a relevant URL just because it appears in a different
  knowledge entry.

==================================================
GENERAL RULES
==================================================

- Answer the user's question directly and concisely.
- Do NOT ask follow-up or clarification questions.
- Do NOT ask for the user's city for general telehealth booking.
- If the context contains useful partial information, use it.
- Clearly distinguish between confirmed and unavailable information.
- Use the same language as the user (Arabic -> Arabic, English -> English).
- Do not mention the knowledge base, context, embeddings, retrieval,
  RAG, or AI system.

Context:

{context}

User question:

{question}
"""

    result = _call_claude(prompt, max_tokens=600)

    if result:
        return result

    return generate_fallback_answer(question, "RELATED_BUT_UNKNOWN")


# ----------------------------------------------------------------------
# 4) Relevance check
# ----------------------------------------------------------------------

def check_context_relevance(
    question: str,
    context: str,
):
    prompt = f"""
Determine whether the provided context contains useful information
that can be used to answer the user's question.

The question may contain spelling mistakes or dialect (e.g. "موقكم" =
"موقعكم", "رقمكن" = "رقمكم"). Judge by the intended meaning.

Return ONLY one of these two values:

ANSWERABLE
NOT_ANSWERABLE

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
information for the user's question.

IMPORTANT BUSINESS CONTEXT:

Hakeem Care provides remote telehealth consultations.
Patients can book doctors remotely and attend consultations online.
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

    result = _call_claude(prompt, max_tokens=20)

    if result:
        result = result.strip().upper()

        if result in {"ANSWERABLE", "NOT_ANSWERABLE"}:
            return result

    return "NOT_ANSWERABLE"


# ----------------------------------------------------------------------
# 5) Knowledge gap classification
# ----------------------------------------------------------------------

def classify_knowledge_gap(
    question: str,
    context: str,
):
    prompt = f"""
Classify the user's question into exactly one of these three categories:

RELATED_BUT_UNKNOWN
MEDICAL_ADVICE
OUT_OF_SCOPE

RELATED_BUT_UNKNOWN:
The question is related to Hakeem Care, its services, products,
doctors, consultations, laboratories, prescriptions, payments,
appointments, contact details, or website, but the available context
does not contain enough information to answer it.

MEDICAL_ADVICE:
The user describes symptoms, asks for a diagnosis, asks which
medication/dose/treatment to take, or asks any personal medical
question that requires a doctor's judgment.

OUT_OF_SCOPE:
The question is clearly unrelated to Hakeem Care, its services, or
the healthcare domain (e.g. sports, cooking, politics, general trivia).

The question may contain spelling mistakes or dialect (e.g. "موقكم" =
"موقعكم"). Judge by the intended meaning.

Examples:

Question: What doctors are available today?
Result: RELATED_BUT_UNKNOWN

Question: وين موقكم
Result: RELATED_BUT_UNKNOWN

Question: عندي صداع من يومين ايش اخذ؟
Result: MEDICAL_ADVICE

Question: How many ants are there in the world?
Result: OUT_OF_SCOPE

IMPORTANT BUSINESS CONTEXT:
Hakeem Care provides remote telehealth consultations. A missing city
or location does NOT make a telehealth question OUT_OF_SCOPE.
Do not classify a question as OUT_OF_SCOPE simply because the exact
answer is missing.

Return ONLY one of:

RELATED_BUT_UNKNOWN
MEDICAL_ADVICE
OUT_OF_SCOPE

Context:

{context}

User question:

{question}
"""

    result = _call_claude(prompt, max_tokens=30)

    if result:
        result = result.strip().upper()

        if result in {"RELATED_BUT_UNKNOWN", "MEDICAL_ADVICE", "OUT_OF_SCOPE"}:
            return result

    # If we are unsure, treat it as related so the user is sent to the
    # website instead of being told the question is out of scope.
    return "RELATED_BUT_UNKNOWN"


# ----------------------------------------------------------------------
# 6) Deterministic fallback (no LLM -> no hallucination)
# ----------------------------------------------------------------------

def generate_fallback_answer(question: str, gap_type: str = "RELATED_BUT_UNKNOWN") -> str:
    """
    Safe, fixed responses used when the knowledge base cannot answer.
    Always points the user to the Hakeem Care website.
    gap_type: RELATED_BUT_UNKNOWN | MEDICAL_ADVICE | OUT_OF_SCOPE
    """

    arabic = _is_arabic(question)

    if gap_type == "MEDICAL_ADVICE":
        if arabic:
            return (
                "عذرًا، ما أقدر أعطيك نصيحة طبية أو تشخيص 🌷\n"
                "تقدر تستشير طبيب مباشرة من خلال تطبيق أو موقع حكيم كير:\n"
                "- استشارة عاجلة بـ 15 ريال\n"
                "- أو احجز موعد مع طبيب في التخصص المناسب\n"
                f"{WEBSITE_URL}"
            )
        return (
            "Sorry, I can't provide medical advice or a diagnosis 🌷\n"
            "You can consult a doctor directly through the Hakeem Care app "
            "or website (urgent consultation is 15 SAR):\n"
            f"{WEBSITE_URL}"
        )

    if gap_type == "OUT_OF_SCOPE":
        if arabic:
            return (
                "أنا مساعد حكيم كير، وأقدر أساعدك في الأسئلة عن خدماتنا "
                "(الاستشارات، الحجوزات، المختبر، الأسعار...) 🌷\n"
                f"وللمزيد من المعلومات زر موقعنا: {WEBSITE_URL}"
            )
        return (
            "I'm the Hakeem Care assistant and can help with questions about "
            "our services (consultations, bookings, labs, prices...) 🌷\n"
            f"For more information, visit: {WEBSITE_URL}"
        )

    # RELATED_BUT_UNKNOWN
    if arabic:
        return (
            "ما لقيت معلومة مؤكدة عن هذا السؤال حاليًا 🌷\n"
            f"تقدر تتأكد من موقع حكيم كير للحصول على أدق المعلومات:\n{WEBSITE_URL}"
        )
    return (
        "I couldn't find confirmed information about this right now 🌷\n"
        f"Please check the Hakeem Care website for the most accurate details:\n{WEBSITE_URL}"
    )