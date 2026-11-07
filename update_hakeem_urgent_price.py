import sys
from sqlalchemy import text
from app.database.connection import engine
from app.embeddings.voyage import create_embeddings

CLIENT_ID = 1

# Records confirmed to describe the urgent consultation price.
TARGET_IDS = [409, 613, 619, 620, 621, 626, 627, 629, 642, 643]

REPLACEMENTS = [
    ("15 ريال سعودي", "20 ريال سعودي"),
    ("15 ريالًا", "20 ريالًا"),
    ("15 ريالاً", "20 ريالًا"),
    ("15 ريال", "20 ريال"),
    ("15 SAR", "20 SAR"),
]

apply_changes = "--apply" in sys.argv

with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT id, content
            FROM knowledge
            WHERE client_id = :client_id
              AND id = ANY(:ids)
            ORDER BY id
        """),
        {"client_id": CLIENT_ID, "ids": TARGET_IDS},
    ).fetchall()

found_ids = {row.id for row in rows}
missing = set(TARGET_IDS) - found_ids

if missing:
    raise RuntimeError(
        f"Aborting: expected records not found: {sorted(missing)}"
    )

updates = []

for row in rows:
    old_content = row.content
    new_content = old_content

    for old, new in REPLACEMENTS:
        new_content = new_content.replace(old, new)

    if new_content != old_content:
        updates.append((row.id, old_content, new_content))

print(f"Records inspected: {len(rows)}")
print(f"Records requiring changes: {len(updates)}")

for record_id, old, new in updates:
    print(f"\n--- Knowledge ID {record_id} ---")
    print("BEFORE:", old[:350])
    print("AFTER: ", new[:350])

if not updates:
    print("No matching price text found. Nothing changed.")
    sys.exit(0)

if not apply_changes:
    print("\nPreview only. No database changes made.")
    print("Review the output, then run with --apply.")
    sys.exit(0)

# Generate embeddings for the updated document text.
embeddings = create_embeddings(
    [new for _, _, new in updates],
    input_type="document",
)

if len(embeddings) != len(updates):
    raise RuntimeError("Embedding count mismatch; aborting.")

if any(len(e) != 1024 for e in embeddings):
    raise RuntimeError("Invalid embedding dimensions; aborting.")

# Update content and embedding together in one transaction.
with engine.begin() as conn:
    for (record_id, old, new), embedding in zip(updates, embeddings):
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
                f"Update failed for ID {record_id}; transaction aborted."
            )

print(f"\nSuccessfully updated {len(updates)} knowledge records.")
