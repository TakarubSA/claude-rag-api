from sqlalchemy import text
from app.database.connection import engine
from app.embeddings.voyage import create_embeddings

with engine.connect() as conn:
    row = conn.execute(
        text("SELECT content FROM knowledge WHERE id = 664 AND client_id = 1")
    ).fetchone()

if not row:
    raise RuntimeError("Record 664 not found")

embeddings = create_embeddings([row.content], input_type="document")

if len(embeddings) != 1 or len(embeddings[0]) != 1024:
    raise RuntimeError("Invalid embedding result")

with engine.begin() as conn:
    result = conn.execute(
        text("""
            UPDATE knowledge
            SET embedding = CAST(:embedding AS vector)
            WHERE id = 664 AND client_id = 1
        """),
        {"embedding": str(embeddings[0])},
    )
    if result.rowcount != 1:
        raise RuntimeError(f"Expected 1 updated row, got {result.rowcount}")

print("Successfully updated embedding for knowledge ID 664 only.")
