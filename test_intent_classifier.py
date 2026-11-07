import json
import time
import argparse
from collections import Counter, defaultdict

from app.services.intent import classify_intent

QUESTIONS_FILE = "intent_v7_questions.json"
RESULTS_FILE = "intent_structured_smoke.jsonl"


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def select_questions(questions, per_category=5):
    grouped = defaultdict(list)

    for q in questions:
        grouped[q["category"]].append(q)

    selected = []

    for category, items in grouped.items():
        if len(items) <= per_category:
            selected.extend(items)
            continue

        # Spread samples across the category instead of
        # selecting only the first wording variants.
        indexes = [
            round(i * (len(items) - 1) / (per_category - 1))
            for i in range(per_category)
        ] if per_category > 1 else [0]

        selected.extend(items[i] for i in indexes)

    return selected


def run(questions):
    results = []

    for index, q in enumerate(questions, 1):
        started = time.time()

        result = {
            **q,
            "actual_intent": "",
            "actual_action": "",
            "reason": "",
            "latency": None,
            "error": "",
        }

        try:
            decision = classify_intent(
                message=q["prompt"],
                conversation_context="",
            )

            result["actual_intent"] = decision["intent"]
            result["actual_action"] = decision["action"]
            result["reason"] = decision["reason"]

        except Exception as exc:
            result["error"] = str(exc)

        result["latency"] = round(time.time() - started, 2)
        results.append(result)

        print(
            f"[{index}/{len(questions)}] "
            f"{q['category']} | "
            f"{result['actual_intent']} | "
            f"{result['actual_action']} | "
            f"{result['latency']}s"
        )

        time.sleep(0.2)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    valid = [r for r in results if not r["error"]]

    correct = [
        r for r in valid
        if r["expected_action"] == r["actual_action"]
    ]

    print("\n========== STRUCTURED INTENT REPORT ==========")
    print("Total:", len(results))
    print("Successful:", len(valid))
    print("Failed:", len(results) - len(valid))

    if valid:
        print(
            "Action accuracy:",
            f"{len(correct) / len(valid) * 100:.1f}%"
        )

    print("\nIntent distribution:")
    print(dict(Counter(r["actual_intent"] for r in valid)))

    print("\nAction distribution:")
    print(dict(Counter(r["actual_action"] for r in valid)))

    print("\nWrong actions:")
    for r in valid:
        if r["expected_action"] != r["actual_action"]:
            print(
                r["category"],
                "| Expected:", r["expected_action"],
                "| Actual:", r["actual_action"],
                "| Intent:", r["actual_intent"],
                "|", r["prompt"],
            )

    latencies = [r["latency"] for r in valid]

    if latencies:
        print("\nAverage latency:", round(sum(latencies) / len(latencies), 2))

    print("\nResults saved to:", RESULTS_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions()
    selected = select_questions(
        questions,
        per_category=args.per_category,
    )

    print("Selected questions:", len(selected))
    print("Categories:", dict(Counter(q["category"] for q in selected)))

    run(selected)
