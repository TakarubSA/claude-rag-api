from pathlib import Path
from datetime import datetime
import shutil

path = Path("app/llm/claude.py")
source = path.read_text(encoding="utf-8")

anchor = """The answer does NOT need to contain every possible detail.

==================================================
NOT_ANSWERABLE
"""

replacement = """The answer does NOT need to contain every possible detail.

==================================================
POLICY-BASED ANSWERS
==================================================

IMPORTANT: A useful answer does NOT always require disclosing
the exact information requested.

Return ANSWERABLE when the retrieved knowledge or client
configuration contains a relevant business policy that:

- Explicitly restricts or prohibits disclosing the requested information.
- Provides an approved alternative action, channel, or process
  for the customer.
- Explains how the customer's request should be handled.

The policy itself is sufficient evidence for a useful response.
Do NOT require the prohibited value (such as a price) to be present.

Example:

Question:
كم سعر باقة التحاليل؟

Retrieved knowledge:
The business does not disclose laboratory test or package prices
in chat. Customers must use the official application or website
to view current prices.

Correct result:
ANSWERABLE

Reason:
The policy provides a clear and approved way to handle the request,
even though the exact price is intentionally not disclosed.

IMPORTANT DISTINCTION:
A price belonging to a different service does not answer the question.
For example, a fee for interpreting laboratory results is not the
price of an individual laboratory test or a laboratory package.

Only consider a policy answerable when its scope actually applies
to the user's request. Do not apply unrelated business policies.

==================================================
NOT_ANSWERABLE
"""

count = source.count(anchor)

if count != 1:
    raise RuntimeError(
        f"Expected exactly one insertion point, found {count}. No changes made."
    )

backup = path.with_name(
    f"claude.py.backup_evaluator_{datetime.now():%Y%m%d_%H%M%S}"
)

shutil.copy2(path, backup)
path.write_text(source.replace(anchor, replacement), encoding="utf-8")

print("Evaluator prompt updated successfully.")
print(f"Backup: {backup}")
