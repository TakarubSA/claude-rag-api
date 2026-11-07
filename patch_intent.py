from pathlib import Path
import shutil

chat = Path("app/routes/chat.py")
intent = Path("app/services/intent.py")

# Backup
shutil.copy2(chat, "app/routes/chat.py.bak")
shutil.copy2(intent, "app/services/intent.py.bak")

code = chat.read_text()

anchor = '''def is_lab_reading_request(message: str) -> bool:
    normalized = " ".join((message or "").lower().split())
    return any(phrase in normalized for phrase in LAB_READING_PHRASES)
'''

addition = '''

RESCHEDULE_ACTIONS = (
    "تغيير موعد",
    "تغير موعد",
    "تعديل موعد",
    "تعدلون موعد",
    "تعديل تاريخ",
    "اعادة جدولة",
    "نقل موعد",
    "تاجيل موعد",
    "تقديم موعد",
    "تغيير وقت",
    "تعديل وقت",
    "ااخر موعد",
    "اقدم موعد",
    "موعد لا يناسبني",
    "الموعد لا يناسبني",
    "موعدي لا يناسبني",
)

SCHEDULE_SERVICES = (
    "استشار",
    "زيارة",
    "التمريض",
    "الممرضة",
    "سحب الدم",
    "سحب العينة",
    "المنزلية",
)


def is_reschedule_request(message: str) -> bool:
    normalized = " ".join((message or "").lower().split())

    # Normalize common Arabic letter variations.
    normalized = (
        normalized
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )

    has_action = any(
        phrase in normalized for phrase in RESCHEDULE_ACTIONS
    )
    has_service = any(
        phrase in normalized for phrase in SCHEDULE_SERVICES
    )

    return has_action and has_service
'''

if "def is_reschedule_request(" not in code:
    if anchor not in code:
        raise SystemExit("Could not find lab detection function. No changes made.")
    code = code.replace(anchor, anchor + addition)

old_call = '''    intent = classify_intent(
        message=message,
        conversation_context=conversation_context,
    )
'''

new_call = '''    if (
        is_lab_reading_request(message)
        or is_reschedule_request(message)
    ):
        intent = "handoff"
        logger.info(
            "Deterministic handoff detected: client_id=%s",
            client.id,
        )
    else:
        intent = classify_intent(
            message=message,
            conversation_context=conversation_context,
        )
'''

if old_call in code:
    code = code.replace(old_call, new_call, 1)
elif "Deterministic handoff detected:" not in code:
    raise SystemExit("Could not find classify_intent call. No changes made.")

chat.write_text(code)

# Update intent prompt
prompt = intent.read_text()

old_reply = "- Questions about how to book or use a service."
new_reply = '''- Questions about how to book or use a service.
- A successful payment where the booking was not completed is
  normally a reply, not a handoff. Explain the next steps and
  do not transfer unless the user explicitly requests an agent
  or a separate business rule requires human intervention.'''

if new_reply not in prompt:
    if old_reply not in prompt:
        raise SystemExit("Could not find reply prompt section.")
    prompt = prompt.replace(old_reply, new_reply, 1)

intent.write_text(prompt)

print("Intent patch applied successfully.")
print("Backups created:")
print(" - app/routes/chat.py.bak")
print(" - app/services/intent.py.bak")
