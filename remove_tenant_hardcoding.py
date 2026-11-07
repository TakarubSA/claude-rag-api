from pathlib import Path
from datetime import datetime
import shutil

intent_path = Path("app/services/intent.py")
repo_path = Path("app/knowledge/repository.py")

intent = intent_path.read_text(encoding="utf-8")
repo = repo_path.read_text(encoding="utf-8")

changes = [
    (
        intent,
        "You are the intent-understanding and routing engine for Hakeem Care.",
        "You are the intent-understanding and routing engine for the current tenant.",
        "Intent prompt identity",
    ),
    (
        intent,
        """13. Do not invent Hakeem Care policies, prices, service availability,
     or customer-specific facts. Those must come from the knowledge base
     in the answer-generation stage.""",
        """13. Do not assume or invent tenant-specific policies, prices,
     service availability, or customer-specific facts. These must be
     determined from the current tenant's configuration and knowledge
     during the answer-generation stage.""",
        "Intent business policy",
    ),
    (
        repo,
        """# Words that express the question/intent rather than the subject being asked about.
# Removing these from lexical search prevents a query such as "كم سعر مونجارو؟"
# from matching every knowledge entry that happens to mention a price.""",
        """# Generic question and intent words that should not dominate lexical search.
# This keeps broad wording from matching unrelated knowledge entries.""",
        "Repository comments",
    ),
    (
        repo,
        """            # Common Arabic spelling variants for the same product name.
            if t in {"منجارو", "مونجارو"}:
                out.extend(["منجارو", "مونجارو"])
            else:
                out.append(t)""",
        """            out.append(t)""",
        "Product-specific spelling normalization",
    ),
]

for source, old, new, label in changes:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence, found {count}. "
            "No files changed."
        )

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for path in (intent_path, repo_path):
    backup = path.with_name(f"{path.name}.backup_generic_{timestamp}")
    shutil.copy2(path, backup)
    print(f"Backup: {backup}")

for path in (intent_path, repo_path):
    source = path.read_text(encoding="utf-8")

    for original, old, new, label in changes:
        if original == source:
            source = source.replace(old, new, 1)

    path.write_text(source, encoding="utf-8")

print("\nTenant-specific code references removed.")
