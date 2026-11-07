import requests
from test_10000_prompts import API_URL, TOKEN

cases = [
    {
        "name": "Refund policy",
        "message": "كم مدة استرداد المبلغ؟",
        "expected": "reply، والإجابة حسب سياسة يوم عمل واحد"
    },
    {
        "name": "Lab test pricing",
        "message": "كم سعر تحليل CBC؟",
        "expected": "reply، توجيه العميل للتطبيق أو الموقع دون اختراع سعر"
    },
    {
        "name": "Refund execution",
        "message": "أريد استرداد رسوم الخدمة التي دفعتها.",
        "expected": "handoff"
    },
    {
        "name": "Ambiguous request",
        "message": "أحتاج مساعدة في طلبي.",
        "expected": "reply مع سؤال توضيحي"
    },
    {
        "name": "Contextual follow-up",
        "message": "كم سعرها؟",
        "context": (
            "user: ما سعر الاستشارة العاجلة؟\n"
            "assistant: أستطيع مساعدتك في معرفة سعر الاستشارة العاجلة."
        ),
        "expected": "reply مع استرجاع سعر الاستشارة العاجلة من المعرفة"
    },
]

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

for i, case in enumerate(cases, 1):
    payload = {
        "message": case["message"],
        "conversation_id": f"smoke-test-{i}",
        "reset_conversation": True,
    }

    if case.get("context"):
        payload["conversation_context"] = case["context"]

    print(f"\n{'=' * 55}")
    print(f"TEST {i}: {case['name']}")
    print("Expected:", case["expected"])

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=90,
        )

        print("HTTP:", response.status_code)

        data = response.json()
        print("Action:", data.get("action"))
        print("Status:", data.get("status"))
        print("Answer:", data.get("answer"))

    except Exception as e:
        print("ERROR:", str(e))

print("\nSMOKE TEST FINISHED")
