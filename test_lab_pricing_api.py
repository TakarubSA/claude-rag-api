import re
import requests

from test_10000_prompts import API_URL, TOKEN

if not TOKEN:
    raise RuntimeError("RAG_API_KEY is missing from the environment.")

tests = [
    ("CBC", "كم سعر تحليل CBC؟"),
    ("HbA1c", "بكم تحليل السكر التراكمي؟"),
    ("Packages", "كم سعر باقة التحاليل؟"),
]

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

for name, question in tests:
    print("\n" + "=" * 75)
    print(f"TEST: {name}")
    print("QUESTION:", question)

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "message": question,
                "conversation_id": f"lab-pricing-{name}",
                "reset_conversation": True,
            },
            timeout=120,
        )

        print("HTTP:", response.status_code)
        response.raise_for_status()

        data = response.json()
        answer = data.get("answer", "")

        print("ACTION:", data.get("action"))
        print("STATUS:", data.get("status"))
        print("ANSWER:")
        print(answer)

        price_pattern = r"\d+(?:[.,]\d+)?\s*(?:ريال|ريالات|SAR)"
        prices = re.findall(price_pattern, answer, re.IGNORECASE)

        if prices:
            print("WARNING: POSSIBLE PRICE DISCLOSURE:", prices)
        else:
            print("PRICE CHECK: No numeric price detected.")

    except Exception as error:
        print("TEST ERROR:", str(error))

print("\nALL TESTS FINISHED")
