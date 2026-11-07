import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from app.database.connection import engine

CLIENT_ID = 1
APPLY = "--apply" in sys.argv

MATCH = "do not assume that prices vary"

NEW_RULE = (
    "When a customer asks for the price of an individual laboratory test "
    "or a laboratory package, never disclose, estimate, or imply any price. "
    "Do not state or speculate that prices differ based on package contents, "
    "type, components, or any other factor unless that exact claim is "
    "explicitly documented in this client's knowledge or configuration. "
    "Do not add pricing explanations, assumptions, or follow-up questions. "
    "For Arabic messages, reply only with: "
    "'لا نعرض أسعار التحاليل الفردية أو باقات التحاليل داخل المحادثة. "
    "للاطلاع على الأسعار الحالية واختيار المطلوب، يرجى استخدام تطبيق "
    "حكيم كير أو الموقع الرسمي https://hakeemcare.com/.' "
    "For other languages, provide the same meaning in the customer's language. "
    "Do not add unrelated claims or offers."
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

if row["config_type"] not in {"json", "jsonb"}:
    raise RuntimeError("Unexpected ai_config database type.")

config = json.loads(row["config_text"])
rules = config.get("response_rules")

if not isinstance(rules, list):
    raise RuntimeError("response_rules must be a list.")

matches = [
    i for i, rule in enumerate(rules)
    if isinstance(rule, str) and MATCH in rule.lower()
]

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one existing rule to replace, found {len(matches)}."
    )

index = matches[0]
old_rule = rules[index]
new_rules = list(rules)
new_rules[index] = NEW_RULE

print("Client:", row["name"])
print("Response rules before:", len(rules))
print("Response rules after:", len(new_rules))
print("\nOLD RULE:\n", old_rule)
print("\nNEW RULE:\n", NEW_RULE)

if not APPLY:
    print("\nPREVIEW ONLY — database unchanged.")
    print("Run with --apply after reviewing.")
    sys.exit(0)

config["response_rules"] = new_rules

backup = Path(
    f"/root/hakeem_ai_config_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
)
backup.write_text(
    json.dumps(json.loads(row["config_text"]), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with engine.begin() as conn:
    result = conn.execute(text(f"""
        UPDATE clients
        SET ai_config = CAST(:new_config AS {row["config_type"]})
        WHERE id = :id
          AND ai_config::text = :old_config
    """), {
        "id": CLIENT_ID,
        "old_config": row["config_text"],
        "new_config": json.dumps(config, ensure_ascii=False),
    })

    if result.rowcount != 1:
        raise RuntimeError("Concurrent update detected; no overwrite performed.")

print("\nUpdated successfully.")
print("Backup:", backup)
