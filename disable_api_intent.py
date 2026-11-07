from pathlib import Path

path = Path("app/services/intent.py")
content = path.read_text(encoding="utf-8")

old = 'VALID_INTENTS = {"reply", "api", "handoff"}'
new = 'VALID_INTENTS = {"reply", "handoff"}'

if old not in content:
    raise SystemExit("ERROR: لم أجد تعريف VALID_INTENTS المتوقع. لم يتم تعديل الملف.")

content = content.replace(old, new, 1)

old_prompt = "Return exactly one action:\nreply, api, or handoff."
new_prompt = "Return exactly one action:\nreply or handoff."

if old_prompt in content:
    content = content.replace(old_prompt, new_prompt, 1)

path.write_text(content, encoding="utf-8")

print("Updated VALID_INTENTS successfully.")
print("\nRemaining references to api:")
for number, line in enumerate(content.splitlines(), 1):
    if "api" in line.lower():
        print(f"{number}: {line}")
