from pathlib import Path

path = Path("app/services/intent.py")
content = path.read_text(encoding="utf-8")

replacements = [
    (
        "    - api: Execute a requested external or live operation.\n",
        ""
    ),
    (
        """api:
Use api only when the user is requesting an actual external
or live operation that the system needs to execute, such as
checking a live order status, retrieving a transaction status,
or performing a supported booking operation.

Do not use api merely because the user asks how a service works,
asks about its price, or wants information about availability.

Only select api when the requested operation is supported
by the system.

""",
        ""
    ),
    (
        """3. A supported external operation explicitly requested by
   the user: api.
4. Otherwise: reply.""",
        """3. Otherwise: reply."""
    ),
    (
        "Return exactly one word:\nreply, api, or handoff.",
        "Return exactly one word:\nreply or handoff."
    ),
]

for old, new in replacements:
    count = content.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: Expected exactly one match, found {count}:\n{old}"
        )
    content = content.replace(old, new, 1)

# تأكد أن التصنيف المسموح لا يحتوي api
if 'VALID_INTENTS = {"reply", "api", "handoff"}' in content:
    raise SystemExit("ERROR: VALID_INTENTS still contains api.")

path.write_text(content, encoding="utf-8")

print("SUCCESS: Removed api from intent instructions.")
print("\nRemaining api references:")
for number, line in enumerate(content.splitlines(), 1):
    if "api" in line.lower():
        print(f"{number}: {line}")
