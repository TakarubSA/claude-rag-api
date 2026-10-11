import argparse
from collections import Counter

from sqlalchemy import text

from app.database.connection import engine
from app.embeddings.voyage import create_embeddings


EXPECTED_DATABASE = "ragdb"
EXPECTED_HOSTS = {"localhost", "127.0.0.1", "::1"}
EXPECTED_DIMENSIONS = 1024
BATCH_SIZE = 8


def verify_local_database():
    """Prevent accidentally re-embedding a production database."""
    host = engine.url.host
    database = engine.url.database

    print(f"Database host: {host}")
    print(f"Database name: {database}")

    if host not in EXPECTED_HOSTS:
        raise RuntimeError(
            "Refusing to continue: database host is not localhost."
        )

    if database != EXPECTED_DATABASE:
        raise RuntimeError(
            f"Refusing to continue: expected {EXPECTED_DATABASE}."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update embeddings. Default is a read-only preview.",
    )
    args = parser.parse_args()

    verify_local_database()

    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT id, client_id, content
                FROM knowledge
                ORDER BY id
            """)
        ).fetchall()

    if not rows:
        print("No knowledge records found.")
        return

    counts = Counter(row.client_id for row in rows)

    print(f"Total records: {len(rows)}")
    print("Records by client:")

    for client_id, count in sorted(counts.items()):
        print(f"  Client {client_id}: {count}")

    if not args.apply:
        print()
        print("Preview only. No embeddings have been changed.")
        print("Back up the database, then run with --apply.")
        return

    print()
    print("Starting local embedding migration...")
    print("Keep the process running until it finishes.")

    updated = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        contents = [
            row.content or ""
            for row in batch
        ]

        embeddings = create_embeddings(
            contents,
            batch_size=BATCH_SIZE,
            input_type="document",
        )

        if len(embeddings) != len(batch):
            raise RuntimeError(
                "Embedding count mismatch; stopping migration."
            )

        for embedding in embeddings:
            if len(embedding) != EXPECTED_DIMENSIONS:
                raise RuntimeError(
                    f"Expected {EXPECTED_DIMENSIONS} dimensions, "
                    f"got {len(embedding)}."
                )

        # Each batch is committed atomically. Existing records are updated,
        # not inserted or deleted.
        with engine.begin() as connection:
            for row, embedding in zip(batch, embeddings):
                connection.execute(
                    text("""
                        UPDATE knowledge
                        SET embedding = CAST(:embedding AS vector)
                        WHERE id = :id
                          AND client_id = :client_id
                    """),
                    {
                        "id": row.id,
                        "client_id": row.client_id,
                        "embedding": str(embedding),
                    },
                )

        updated += len(batch)
        print(f"Updated {updated}/{len(rows)} records")

    print()
    print("Re-embedding completed successfully.")
    print(f"Updated records: {updated}")


if __name__ == "__main__":
    main()
