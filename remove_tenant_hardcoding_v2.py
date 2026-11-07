from pathlib import Path
from datetime import datetime
import shutil

replacements = {
    Path("app/services/intent.py"): [
        (
            "You are the intent-understanding and routing engine for Hakeem Care.",
            "You are the intent-understanding and routing engine for the current tenant.",
        ),
        (
            "13. Do not invent Hakeem Care policies, prices, service availability,\n"
            "     or customer-specific facts. Those must come from the knowledge base\n"
            "     in the answer-generation stage.",
            "13. Do not assume or invent tenant-specific policies, prices,\n"
            "     service availability, or customer-specific facts. These must be\n"
            "     determined from the current tenant's configuration and knowledge.",
        ),
    ],
    Path("app/knowledge/repository.py"): [
        (
            '# Words that express the question/intent rather than the subject being asked about.\n'
            '# Removing these from lexical search prevents a query such as "كم سعر مونجارو؟"\n'
            '# from matching every knowledge entry that happens to mention a price.',
            "# Generic question and intent words that should not dominate lexical search.\n"
            "# This prevents broad wording from matching unrelated knowledge entries.",
        ),
        (
            '            # Common Arabic spelling variants for the same product name.\n'
            '            if t in {"منجارو", "مونجارو"}:\n'
            '                out.extend(["منجارو", "مونجارو"])\n'
            '            else:\n'
            '                out.append(t)',
            "            out.append(t)",
        ),
    ],
}

updated_files = {}

for path, pairs in replacements.items():
    source = path.read_text(encoding="utf-8")

    for old, new in pairs:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"{path}: expected exactly one occurrence, found {count}."
            )
        source = source.replace(old, new, 1)

    updated_files[path] = source

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for path, updated in updated_files.items():
    backup = path.with_name(f"{path.name}.backup_generic_{timestamp}")
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    print(f"Updated: {path}")
    print(f"Backup: {backup}")

print("All requested replacements completed.")
