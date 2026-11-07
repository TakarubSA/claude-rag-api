from pathlib import Path
from datetime import datetime
import shutil

intent_path = Path("app/services/intent.py")
repo_path = Path("app/knowledge/repository.py")

intent = intent_path.read_text(encoding="utf-8")
repo = repo_path.read_text(encoding="utf-8")

# Intent prompt: remove tenant identity.
identity = "You are the intent-understanding and routing engine for Hakeem Care."
if intent.count(identity) != 1:
    raise RuntimeError("Intent identity check failed.")

intent = intent.replace(
    identity,
    "You are the intent-understanding and routing engine for the current tenant.",
    1,
)

# Intent prompt: replace tenant-specific policy wording.
intent_lines = intent.splitlines(keepends=True)

policy_indexes = [
    i for i, line in enumerate(intent_lines)
    if "13. Do not invent Hakeem Care policies, prices, service availability," in line
]

if len(policy_indexes) != 1:
    raise RuntimeError("Intent policy location check failed.")

i = policy_indexes[0]
block = "".join(intent_lines[i:i + 3])

if (
    "or customer-specific facts. Those must come from the knowledge base" not in block
    or "in the answer-generation stage." not in block
):
    raise RuntimeError("Intent policy content check failed.")

indent = intent_lines[i][:len(intent_lines[i]) - len(intent_lines[i].lstrip())]

intent_lines[i:i + 3] = [
    indent + "13. Do not assume or invent tenant-specific policies, prices,\n",
    indent + "    service availability, or customer-specific facts. These must be\n",
    indent + "    determined from the current tenant's configuration and knowledge.\n",
]

intent = "".join(intent_lines)

# Repository: replace the product-specific comment.
repo_lines = repo.splitlines(keepends=True)

comment_indexes = [
    i for i, line in enumerate(repo_lines)
    if line.startswith("# Words that express the question/intent")
]

if len(comment_indexes) != 1:
    raise RuntimeError("Repository comment location check failed.")

i = comment_indexes[0]
comment_block = "".join(repo_lines[i:i + 3])

if (
    "Removing these from lexical search" not in comment_block
    or "every knowledge entry" not in comment_block
):
    raise RuntimeError("Repository comment content check failed.")

repo_lines[i:i + 3] = [
    "# Generic question and intent words that should not dominate lexical search.\n",
    "# This prevents broad wording from matching unrelated knowledge entries.\n",
]

repo = "".join(repo_lines)

# Repository: remove the product-specific spelling exception.
variant_indexes = [
    i for i, line in enumerate(repo_lines)
    if "Common Arabic spelling variants for the same product name." in line
]

if len(variant_indexes) != 1:
    raise RuntimeError("Product spelling rule location check failed.")

i = variant_indexes[0]
variant_block = "".join(repo_lines[i:i + 5])

if (
    'if t in {"منجارو", "مونجارو"}:' not in variant_block
    or 'out.extend(["منجارو", "مونجارو"])' not in variant_block
    or "out.append(t)" not in variant_block
):
    raise RuntimeError("Product spelling rule content check failed.")

indent = repo_lines[i][:len(repo_lines[i]) - len(repo_lines[i].lstrip())]

repo_lines[i:i + 5] = [
    indent + "out.append(t)\n",
]

repo = "".join(repo_lines)

# All validations have passed. Back up, then write.
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

print("All four tenant-specific code references removed.")
