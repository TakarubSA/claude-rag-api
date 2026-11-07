import re
import sys

from sqlalchemy import text

from app.database.connection import engine
from app.documents.loader import prepare_documents
from app.embeddings.voyage import create_embeddings
from app.knowledge.repository import insert_knowledge_many


CLIENT_ID = 1
APPLY = "--apply" in sys.argv

ANSWER = (
    "لا يعرض المساعد أسعار التحاليل المخبرية الفردية أو باقات التحاليل داخل "
    "المحادثة، ولا يقدم أسعارًا تقديرية أو قديمة. للاطلاع على الأسعار الحالية "
    "واختيار التحاليل المطلوبة، يرجى استخدام تطبيق Hakeem Care أو الموقع الرسمي: "
    "https://hakeemcare.com/. لا تحوّل العميل إلى خدمة العملاء لمجرد الاستفسار "
    "عن أسعار التحاليل أو الباقات. هذه السياسة خاصة بأسعار التحاليل والباقات، "
    "ولا تشمل خدمة تفسير نتائج التحاليل التي تُعامل كخدمة منفصلة."
)

ENTRIES = [
    {
        "question": "كم سعر تحليل CBC؟",
        "answer": ANSWER,
    },
    {
        "question": "كم سعر تحليل مخبري فردي؟",
        "answer": ANSWER,
    },
    {
        "question": "كم أسعار باقات التحاليل؟",
        "answer": ANSWER,
    },
    {
        "question": "كيف أقدر أعرف أسعار التحاليل الحالية أو أحصل على قائمة الأسعار؟",
        "answer": ANSWER,
    },
]


def normalize(value):
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


with engine.connect() as conn:
    client = conn.execute(
        text("SELECT id, name FROM clients WHERE id = :id"),
        {"id": CLIENT_ID},
    ).first()

    if not client:
        raise RuntimeError(f"Client ID {CLIENT_ID} does not exist.")

    if client.name.strip().casefold() != "hakeem care":
        raise RuntimeError(
            f"Safety check failed: client {CLIENT_ID} is {client.name!r}"
        )

    existing = conn.execute(
        text("""
            SELECT id, content
            FROM knowledge
            WHERE client_id = :client_id
              AND (
                    content ILIKE '%أسعار التحاليل%'
                 OR content ILIKE '%باقات التحاليل%'
                 OR content ILIKE '%سعر تحليل%'
                 OR content ILIKE '%التحاليل الفردية%'
              )
            ORDER BY id DESC
            LIMIT 50
        """),
        {"client_id": CLIENT_ID},
    ).fetchall()


prepared = prepare_documents(ENTRIES)

if len(prepared) != len(ENTRIES):
    raise RuntimeError(
        f"Expected {len(ENTRIES)} documents, got {len(prepared)}."
    )

target_texts = {normalize(item) for item in prepared}
existing_exact = {
    normalize(row.content)
    for row in existing
    if normalize(row.content) in target_texts
}
related = [
    row for row in existing
    if normalize(row.content) not in target_texts
]
to_insert = [
    item for item in prepared
    if normalize(item) not in existing_exact
]

print(f"Client: {client.name} (ID: {client.id})")
print(f"Prepared documents: {len(prepared)}")
print(f"Exact existing documents: {len(existing_exact)}")
print(f"Related existing records: {len(related)}")
print(f"Documents to insert: {len(to_insert)}")

if related:
    print("\nRelated records need manual review:")
    for row in related:
        print(f"\nID: {row.id}")
        print(row.content[:700])

    print("\nNo changes made. Review these records before proceeding.")
    sys.exit(2)

if not to_insert:
    print("\nAll target entries already exist. Nothing to insert.")
    sys.exit(0)

print("\nDocuments prepared for insertion:")
for index, content in enumerate(to_insert, 1):
    print(f"\n--- {index} ---\n{content}")

if not APPLY:
    print("\nPREVIEW ONLY: no database changes were made.")
    print("After reviewing, run:")
    print("python3 add_lab_pricing_knowledge.py --apply")
    sys.exit(0)


embeddings = create_embeddings(
    to_insert,
    input_type="document",
)

if len(embeddings) != len(to_insert):
    raise RuntimeError("Embedding count mismatch. Nothing inserted.")

if any(len(embedding) != 1024 for embedding in embeddings):
    raise RuntimeError("Invalid embedding dimensions. Nothing inserted.")

insert_knowledge_many(
    contents=to_insert,
    embeddings=embeddings,
    client_id=CLIENT_ID,
)

print(f"\nSuccessfully inserted {len(to_insert)} knowledge entries.")

with engine.connect() as conn:
    for content in to_insert:
        row = conn.execute(
            text("""
                SELECT id
                FROM knowledge
                WHERE client_id = :client_id
                  AND content = :content
                ORDER BY id DESC
                LIMIT 1
            """),
            {"client_id": CLIENT_ID, "content": content},
        ).first()

        print(f"Verified knowledge ID: {row.id if row else 'NOT FOUND'}")
