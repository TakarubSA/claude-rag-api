import sys

from sqlalchemy import text

from app.database.connection import engine
from app.embeddings.voyage import create_embeddings

CLIENT_ID = 1
APPLY = "--apply" in sys.argv

NEW_214 = """حكيم كير — السياسة العامة لأسعار التحاليل والفحوصات والباقات.

تشمل هذه السياسة الأسئلة التالية:
كم سعر تحليل CBC؟ كم سعر صورة الدم الكاملة؟ بكم تحليل السكر التراكمي HbA1c؟ كم سعر تحليل فيتامين D؟ كم سعر تحليل منفرد؟ كم سعر باقة التحاليل؟ بكم باقات الفحوصات والتحاليل؟

لا يجوز للمساعد ذكر أو تأكيد أي سعر للتحليل المخبري الفردي أو باقة التحاليل داخل المحادثة، حتى لو ظهر سعر في سجل معرفة آخر. لا يقدم أسعارًا تقديرية أو قديمة، ولا يستنتج سعر تحليل من سعر باقة أو العكس.

عند السؤال عن الأسعار، وجّه العميل إلى تطبيق Hakeem Care أو الموقع الرسمي https://hakeemcare.com/ للاطلاع على الأسعار الحالية واختيار التحاليل أو الباقات المطلوبة.

لا تحوّل العميل إلى خدمة العملاء لمجرد الاستفسار عن أسعار التحاليل أو الباقات.

هذه السياسة خاصة بأسعار التحاليل الفردية وباقات التحاليل، ولا تشمل خدمات أخرى مثل تفسير نتائج التحاليل أو رسوم الزيارة المنزلية."""

REPLACEMENTS = {
    638: (
        "لا تخمّن سعرًا غير موجود في المعلومات المتاحة.",
        "لا تذكر أو تؤكد أي سعر للتحليل الفردي أو باقات التحاليل داخل المحادثة، "
        "حتى لو ظهر سعر في معلومات أخرى. وجّه العميل إلى التطبيق أو الموقع "
        "للاطلاع على الأسعار الحالية.",
    ),
    639: (
        "لا تذكر سعرًا محددًا ما لم يكن مؤكدًا في المعلومات المتاحة، "
        "ولا تفترض أن التحاليل المنفردة لها سعر الباقة الشاملة.",
        "لا تذكر أو تؤكد أي سعر للتحليل الفردي أو الباقات داخل المحادثة، "
        "حتى لو ظهر السعر في معلومات أخرى. وجّه العميل إلى التطبيق أو الموقع "
        "للاطلاع على الأسعار الحالية، ولا تفترض أن التحليل الفردي له سعر الباقة الشاملة.",
    ),
}

with engine.connect() as conn:
    client = conn.execute(
        text("SELECT id, name FROM clients WHERE id = :id"),
        {"id": CLIENT_ID},
    ).first()

    if not client or client.name.strip().casefold() != "hakeem care":
        raise RuntimeError("Client verification failed.")

    rows = conn.execute(
        text("""
            SELECT id, content
            FROM knowledge
            WHERE client_id = :client_id
              AND id IN (214, 638, 639)
            ORDER BY id
        """),
        {"client_id": CLIENT_ID},
    ).fetchall()

records = {row.id: row.content for row in rows}

if set(records) != {214, 638, 639}:
    raise RuntimeError("Expected records 214, 638 and 639 were not all found.")

updates = {
    214: (records[214], NEW_214),
}

for record_id, (old_phrase, new_phrase) in REPLACEMENTS.items():
    old_content = records[record_id]

    if old_phrase not in old_content:
        raise RuntimeError(
            f"Expected text not found in knowledge ID {record_id}. Aborting."
        )

    new_content = old_content.replace(old_phrase, new_phrase)

    if new_content == old_content:
        raise RuntimeError(f"No effective change for ID {record_id}.")

    updates[record_id] = (old_content, new_content)

print(f"Client: {client.name} (ID {client.id})")
print("Records to update:", sorted(updates))

for record_id, (old, new) in updates.items():
    print(f"\n{'=' * 25} ID {record_id} {'=' * 25}")
    print("BEFORE:")
    print(old)
    print("\nAFTER:")
    print(new)

if not APPLY:
    print("\nPREVIEW ONLY — no changes made.")
    print("After reviewing, run: python3 update_lab_pricing_policy.py --apply")
    sys.exit(0)

contents = [updates[i][1] for i in sorted(updates)]
embeddings = create_embeddings(contents, input_type="document")

if len(embeddings) != len(contents):
    raise RuntimeError("Embedding count mismatch.")

if any(len(e) != 1024 for e in embeddings):
    raise RuntimeError("Invalid embedding dimensions.")

with engine.begin() as conn:
    for (record_id, (old, new)), embedding in zip(
        sorted(updates.items()), embeddings
    ):
        result = conn.execute(
            text("""
                UPDATE knowledge
                SET content = :content,
                    embedding = CAST(:embedding AS vector)
                WHERE id = :id
                  AND client_id = :client_id
                  AND content = :old_content
            """),
            {
                "id": record_id,
                "client_id": CLIENT_ID,
                "old_content": old,
                "content": new,
                "embedding": str(embedding),
            },
        )

        if result.rowcount != 1:
            raise RuntimeError(
                f"Concurrent modification or update failure on ID {record_id}."
            )

print("\nSuccessfully updated all three records and their embeddings.")
