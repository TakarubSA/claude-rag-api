import os

import anthropic
from dotenv import load_dotenv


load_dotenv()


client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def rewrite_query(question: str) -> str:
    """
    Rewrite the user's question into a clearer search query.

    This function does NOT answer the user.
    It only improves the query before embedding/retrieval.
    """

    prompt = f"""
You are a query rewriting assistant for Hakeem Care.

Your job is ONLY to rewrite the user's question into a clear search query
for knowledge retrieval.

Do NOT answer the question.
Do NOT add information that is not present in the user's question.

Rules:
- Convert Saudi/Gulf colloquial Arabic into clear Arabic.
- Normalize common spelling variations.
- Preserve the original intent.
- Make very short questions explicit when the intent is obvious.
- Keep important service names and entities.
- Keep the result short.
- Return ONLY the rewritten query.

Examples:

User:
شو رقمكم

Rewritten:
ما هو رقم التواصل؟

User:
شو رقمكن

Rewritten:
ما هو رقم التواصل؟

User:
طيب شو رقمكم

Rewritten:
ما هو رقم التواصل؟

User:
وين موقعكم

Rewritten:
ما هو الموقع الإلكتروني؟

User:
ايش موقع حكيم كير

Rewritten:
ما هو الموقع الإلكتروني لحكيم كير؟

User:
شو ايميلكم

Rewritten:
ما هو البريد الإلكتروني؟

User:
كيف احجز دكتور

Rewritten:
كيف يمكن حجز موعد مع طبيب؟

User:
عندكم دكتور مسالك

Rewritten:
هل تتوفر استشارة في تخصص المسالك البولية؟

User:
كم سعر قراءة التحاليل

Rewritten:
كم تبلغ تكلفة قراءة نتائج التحاليل؟

User question:
{question}
"""

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=80,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            return block.text.strip()

    # Fallback:
    # If Claude doesn't return a text block, keep the original query.
    return question


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

    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            return block.text

    raise RuntimeError("Claude did not return a text response")


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

Examples:

Context:
The official website is:

https://hakeemcare.com

Question:
ما هو موقع حكيم كير؟

Result:
ANSWERABLE


Context:
The website is www.hakeemcare.com.

Patients can access Hakeem Care through the website or application.

Question:
ايش موقع حكيم كير؟

Result:
ANSWERABLE


Context:
Vitamin D test: 99 SAR.

Comprehensive package: 349 SAR.

Lab prices vary by test.

Question:
كم اسعار التحاليل؟

Result:
ANSWERABLE


Context:
Laboratory result interpretation costs 15 SAR.

Question:
كم سعر قراءة التحاليل؟

Result:
ANSWERABLE


Context:
Patients can book doctors remotely through online consultations.

Question:
كيف أحجز دكتور؟

Result:
ANSWERABLE


Context:
The company provides healthcare services.

Question:
كم سعر طبيب القلب غدًا الساعة 8؟

Result:
NOT_ANSWERABLE


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

    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
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

    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text:
            result = block.text.strip().upper()

            if result in {
                "RELATED_BUT_UNKNOWN",
                "OUT_OF_SCOPE",
            }:
                return result

    return "OUT_OF_SCOPE"