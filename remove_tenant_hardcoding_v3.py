from pathlib import Path
from datetime import datetime
import re
import shutil

intent_path = Path("app/services/intent.py")
repo_path = Path("app/knowledge/repository.py")

intent = intent_path.read_text(encoding="utf-8")
repo = repo_path.read_text(encoding="utf-8")

# 1. Generic intent identity
intent, identity_count = re.subn(
    r"You are the intent-understanding and routing engine for Hakeem Care\.",
    "You are the intent-understanding and routing engine for the current tenant.",
    intent,
)

# 2. Generic tenant-policy instruction
policy_pattern = (
    r"(?m)^(?P<i>[ \t]*)13\. Do not invent Hakeem Care policies, prices, service availability,\n"
    r"[ \t]*or customer-specific facts\. Those must come from the knowledge base\n"
    r"[ \t]*in the answer-generation stage\."
)

policy_replacement = (
    r"\g<i>13. Do not assume or invent tenant-specific policies, prices,\n"
    r"     service availability, or customer-specific facts. These must be\n"
    r"     determined from the current tenant's configuration and knowledge."
)

intent, policy_count = re.subn(
    policy_pattern,
    policy_replacement,
    intent,
)

# 3. Remove product-specific comment from repository
comment_pattern = (
    r"(?m)^# Words that express the question/intent rather than the subject being asked about\.\n"
    r"# Removing these from lexical search prevents a query such as \"كم سعر مونجارو\?\"\n"
    r"# from matching every knowledge entry that happens to mention a price\."
)

repo, comment_count = re.subn(
    comment_pattern,
    "# Generic question and intent words that should not dominate lexical search.\n"
    "# This prevents broad wording from matching unrelated knowledge entries.",
    repo,
)

# 4. Remove product-specific spelling exception
variant_pattern = (
    r"(?m)^(?P<i>[ \t]*)# Common Arabic spelling variants for the same product name\.\n"
    r"(?P=i)if t in \{\"منجارو\", \"مونجارو\"\}:\n"
    r"(?P=i)    out\.extend\(\[\"منجارو\", \"مونجارو\"\]\)\n"
    r"(?P=i)else:\n"
    r"(?P=i)    out\.append\(t\)"
)

repo, variant_count = re.subn(
    variant_pattern,
    r"\g<i>out.append(t)",
    repo,
)

checks = {
    "Intent identity": identity_count,
    "Intent policy": policy_count,
    "Repository comment": comment_count,
    "Product spelling rule": variant_count,
}

print("Pre-write validation:")
for label, count in checks.items():
    print(f"- {label}: {count}")

if any(count != 1 for count in checks.values()):
    raise RuntimeError(
        "Validation failed. No files were changed. "
        "Inspect the exact source before retrying."
    )

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for path, updated in (
    (intent_path, intent),
    (repo_path, repo),
):
    backup = path.with_name(f"{path.name}.backup_generic_{timestamp}")
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    print(f"Updated: {path}")
    print(f"Backup: {backup}")

print("\nAll tenant-specific replacements completed.")
