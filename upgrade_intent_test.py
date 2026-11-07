from pathlib import Path

path = Path("test_intent_v2.py")
content = path.read_text(encoding="utf-8")

content = content.replace(
    'RESULTS = "intent_v2_results.jsonl"',
    'RESULTS = "intent_v3_results.jsonl"'
)
content = content.replace(
    'CSV_FILE = "intent_v2_results.csv"',
    'CSV_FILE = "intent_v3_results.csv"'
)

start = content.index("def generate():")
end = content.index("\ndef run(questions):", start)

new_generate = '''def generate():
    questions = []
    counter = 1

    for category, prompts in CASES.items():
        for prompt in prompts:
            variants = [
                prompt,
                f"لو سمحت، {prompt}",
                f"{prompt} ممكن تساعدني؟",
            ]

            for variant in variants:
                questions.append({
                    "id": f"V3-{counter:03d}",
                    "category": category,
                    "prompt": variant,
                    "expected_action": (
                        "handoff" if category == "handoff" else "reply"
                    )
                })
                counter += 1

    if len(questions) != 300:
        raise ValueError(f"Expected 300 questions, got {len(questions)}")

    with open("intent_v3_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print("Generated:", len(questions))
    print("Distribution:", dict(Counter(q["category"] for q in questions)))
    return questions

'''

content = content[:start] + new_generate + content[end:]

path.write_text(content, encoding="utf-8")
print("SUCCESS: Test upgraded to 300 questions.")
