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
provided in the context and the business rules below.


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


IMPORTANT BOOKING AND SERVICE RULE:

If the user's question is about:

- Booking a doctor
- Booking a telehealth consultation
- Starting a treatment
- Accessing a healthcare service
- Using an external service provided by Hakeem Care

do NOT merely explain the service.

Instead:

1. Clearly tell the user what they need to do next.
2. Give the user practical steps based on the available information.
3. Provide the relevant Hakeem Care website or service link when one
   is available.
4. If the service is a remote telehealth service, make it clear that
   the patient can complete the process remotely.
5. Do NOT ask for the user's city or location for general telehealth
   booking.
6. Do NOT tell the user to search for the service themselves if a
   relevant direct link is available.
7. Prefer a direct service URL over a general website URL when a
   specific service URL is available.
8. Never invent or modify a URL.

OBESITY TREATMENT:

If the user's question is specifically about obesity treatment,
weight-loss treatment, or starting obesity treatment through
Hakeem Care, provide this exact URL:

https://hakeemcare.com/patient/obesity-treatment

Copy the URL exactly.

Do NOT use the obesity treatment URL for unrelated services such as
gastroenterology, laboratory services, prescriptions, or other
services unless the user is specifically asking about obesity
treatment.


IMPORTANT LINK RULES:

- Inspect the entire context for relevant URLs.
- If a relevant URL exists in the context, include it in the answer.
- If the user is asking how to book, where to book, or how to access
  a service, provide the relevant link.
- If a specific service URL exists, prefer it over a general website URL.
- Never invent a URL.
- Never modify a URL.
- Copy URLs exactly as provided.
- Do not omit a relevant URL just because it appears in a different
  knowledge entry.
- A URL does not need to appear in the same paragraph as the
  information describing the service.


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
- Tell the user what to do next when the question is asking for an
  action or service.
- Use the same language as the user.
- If the user asks in Arabic, answer in Arabic.
- If the user asks in English, answer in English.
- Do not mention the internal knowledge base, context, embeddings,
  retrieval, RAG, or AI system.


EXAMPLES:

User:
"أبغى أحجز دكتور"

Good answer:
"تقدر تحجز موعد مع الطبيب عن بُعد من خلال حكيم كير. اختر التخصص،
ثم الطبيب والموعد المناسب وأكمل الحجز.

[Relevant Hakeem Care booking link]"

User:
"أبغى علاج للسمنة"

Good answer:
"يمكنك البدء في علاج السمنة من خلال صفحة العلاج المخصصة في حكيم كير:

https://hakeemcare.com/patient/obesity-treatment"

User:
"كيف أبدأ علاج السمنة؟"

Good answer:
"يمكنك البدء من خلال صفحة علاج السمنة في حكيم كير واتباع خطوات
الحجز الموضحة هناك:

https://hakeemcare.com/patient/obesity-treatment"


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

If the question is specifically about obesity treatment and the
question can be answered using the Hakeem Care obesity treatment
service, consider it ANSWERABLE.

Do not mark a question NOT_ANSWERABLE simply because the patient's
city or location is not provided when the question is about remote
telehealth.


IMPORTANT URL RULE:

Inspect the entire context.

If a relevant Hakeem Care website URL or booking URL exists anywhere
in the context, it can be used to answer a question asking where or
how to access the service.

Do not require the URL to appear in the same knowledge entry as the
service description.


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


IMPORTANT BOOKING AND SERVICE RULE:

Questions about:

- Booking doctors
- Telehealth consultations
- Starting treatments
- Accessing Hakeem Care services
- Hakeem Care healthcare services

are within the scope of Hakeem Care.

A missing city or location does NOT make these questions
OUT_OF_SCOPE.

The user should normally be given instructions and a relevant
service link when the required information is available.


IMPORTANT RULES:

- Do not classify a question as OUT_OF_SCOPE simply because
  the exact answer is missing.
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