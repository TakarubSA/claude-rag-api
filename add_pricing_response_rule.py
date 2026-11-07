import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from app.database.connection import engine

CLIENT_ID = 1
APPLY = "--apply" in sys.argv

NEW_RULE = (
    "When asked about individual laboratory test or package prices, "
    "do not assume that prices vary based on package type or included tests. "
    "Do not invent package contents, pricing factors, or explanations for "
    "price differences unless explicitly documented in this client's "
    "knowledge or configuration. Direct customers to the application or "
    "official website to view current prices."
)

with engine.connect() as conn:
    row = conn.execute(text("""
        SELECT id, name, ai_config::text AS config_text,
               pg_typeof(ai_config)::text AS config_type
        FROM clients
        WHERE id = :id
    """), {"id": CLIENT_ID}).mappings().first()

if not row or row["name"].strip().casefold() != "hakeem care":
    raise RuntimeError("Client verification failed.")

config_type = row["config_type"]
if config_type not in {"json", "jsonb"}:
    raise RuntimeError(f"Unexpected ai_config type: {config_type}")

old_text = row["config_text"]
config = json.loads(old_text)

rules = config.get("response_rules", [])
if not isinstance(rules, list):
    raise RuntimeError("response_rules is not a list.")

if NEW_RULE in rules:
    print("Rule already exists. Nothing to do.")
    sys.exit(0)

new_config = dict(config)
new_config["response_rules"] = [*rules, NEW_RULE]

print("Client:", row["name"])
print("Existing response rules:", len(rules))
print("New response rules:", len(new_config["response_rules"]))
print("\nRule to add:\n", NEW_RULE)

if not APPLY:
    print("\nPREVIEW ONLY — database unchanged.")
    print("Run with --apply to save.")
    sys.exit(0)

backup = Path(
    f"/root/hakeem_ai_config_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
)
backup.write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with engine.begin() as conn:
    result = conn.execute(
        text(f"""
            UPDATE clients
            SET ai_config = CAST(:new_config AS {config_type})
            WHERE id = :id
              AND ai_config::text = :old_config
        """),
        {
            "id": CLIENT_ID,
            "old_config": old_text,
            "new_config": json.dumps(new_config, ensure_ascii=False),
        },
    )

    if result.rowcount != 1:
        raise RuntimeError("Concurrent change detected. Update aborted.")

print("\nConfiguration updated successfully.")
print("Backup:", backup)
