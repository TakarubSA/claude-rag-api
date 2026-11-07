
import os
import json
import csv
import time
import requests
from datetime import datetime
from collections import Counter

BASE_URL = "http://127.0.0.1:8000/chat"
API_TOKEN = os.getenv("RAG_API_KEY", "")

INPUT_FILE = "test_prompts.json"
OUTPUT_JSON = "rag_test_results.json"
OUTPUT_CSV = "rag_test_results.csv"

TIMEOUT = 180
DELAY = 1

if not API_TOKEN:
    raise SystemExit("ERROR: Set RAG_API_KEY first")

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}


def ask_rag(question):
    start = time.time()

    response = requests.post(
        BASE_URL,
        headers=HEADERS,
        json={"message": question},
        timeout=TIMEOUT
    )

    elapsed = round(time.time() - start, 2)
    response.raise_for_status()

    return response.json(), elapsed


def save_results(results):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        fields = [
            "id", "category", "prompt", "answer",
            "action", "status", "best_distance",
            "latency", "error"
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in results:
            writer.writerow({
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "answer": item.get("answer", ""),
                "action": item.get("action", ""),
                "status": item.get("status", ""),
                "best_distance": item.get("best_distance"),
                "latency": item.get("latency", 0),
                "error": item.get("error", "")
            })


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        prompts = json.load(f)

    results = []
    total = len(prompts)

    print("\n" + "=" * 55)
    print("       HAKEEM CARE - RAG 100 PROMPTS TEST")
    print("=" * 55)
    print(f"API: {BASE_URL}")
    print(f"Total prompts: {total}\n")

    for index, item in enumerate(prompts, 1):
        question = item["prompt"]

        print(f"[{index}/{total}] {item['id']}")
        print(f"Q: {question}")

        record = {
            "id": item["id"],
            "category": item["category"],
            "prompt": question,
            "evaluation_note": item.get("evaluation_note", ""),
            "tested_at": datetime.now().isoformat()
        }

        try:
            data, latency = ask_rag(question)

            record.update({
                "answer": data.get("answer", ""),
                "action": data.get("action", ""),
                "status": data.get("status", ""),
                "best_distance": data.get("best_distance"),
                "latency": latency,
                "error": ""
            })

            print(f"A: {record['answer']}")
            print(
                f"Status: {record['status']} | "
                f"Distance: {record['best_distance']} | "
                f"Time: {latency}s"
            )

        except Exception as e:
            record.update({
                "answer": "",
                "action": "",
                "status": "REQUEST_FAILED",
                "best_distance": None,
                "latency": 0,
                "error": str(e)
            })

            print(f"ERROR: {e}")

        results.append(record)

        # Save continuously, even if the process stops
        save_results(results)

        print("-" * 55)
        time.sleep(DELAY)

    # Summary
    successful = [
        r for r in results
        if r["status"] != "REQUEST_FAILED"
    ]

    failed = total - len(successful)

    statuses = Counter(r["status"] for r in successful)
    actions = Counter(r["action"] for r in successful)

    avg_latency = (
        sum(r["latency"] for r in successful) / len(successful)
        if successful else 0
    )

    print("\n" + "=" * 55)
    print("              FINAL REPORT")
    print("=" * 55)

    print(f"Total: {total}")
    print(f"API Success: {len(successful)}")
    print(f"API Failed: {failed}")
    print(f"Average Latency: {avg_latency:.2f}s")

    print("\nStatuses:")
    for key, value in statuses.items():
        print(f"  {key}: {value}")

    print("\nActions:")
    for key, value in actions.items():
        print(f"  {key}: {value}")

    print("\nFiles generated:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_CSV}")

    print("\nNote: API success does NOT mean answer correctness.")
    print("Review answers for accuracy and hallucinations.")


if __name__ == "__main__":
    main()
