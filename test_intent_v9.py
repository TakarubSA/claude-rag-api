import json
import csv
import time
import requests
from collections import Counter
from test_10000_prompts import API_URL, TOKEN, DELAY

RESULTS = "intent_v9_results.jsonl"
CSV_FILE = "intent_v9_results.csv"

CASES = {
    "reply": [
        "كم سعر خدمة تفسير نتائج التحاليل؟",
        "كم رسوم الاستشارة العاجلة؟",
        "أين أجد أسعار التحاليل؟",
        "هل تفسير نتائج دلتا برسوم؟",
        "هل حكيم لاب يقدم تفسير النتائج بدون رسوم إضافية؟",
        "ما الخدمات التي تقدمها حكيم كير؟",
        "كيف أستخدم تطبيق حكيم كير؟",
        "ما الفرق بين تفسير التحاليل والاستشارة؟",
        "هل يمكن إرسال تقرير PDF؟",
        "كم تستغرق مدة استرداد المبلغ؟",
        "هل يرجع المبلغ خلال يوم عمل؟",
        "ما سياسة استرداد المبالغ؟",
        "هل يمكن استرداد المبلغ إلى البطاقة؟",
        "هل تختلف رسوم الاسترداد حسب طريقة الدفع؟",
        "كيف أعرف حالة طلبي؟",
        "ما المختبرات المتعاونة مع حكيم كير؟",
        "هل حكيم كير مختبر طبي؟",
        "هل يمكن تفسير نتائج مختبر غير متعاون؟",
        "كيف أرسل نتيجة التحليل؟",
        "ما البيانات المطلوبة لتفسير النتائج؟",
        "هل يمكن تفسير أكثر من تحليل؟",
        "هل يمكن إرسال نتيجة قديمة؟",
        "كيف أحصل على استشارة طبية؟",
        "ما الفرق بين الاستشارة العادية والعاجلة؟",
        "هل خدمة تفسير النتائج متاحة؟",
        "أريد معرفة أسعار الباقات.",
        "هل الأسعار موجودة في التطبيق؟",
        "كيف أطلع على أسعار التحاليل؟",
        "هل رسوم تفسير التحاليل ثابتة؟",
        "كم سعر قراءة نتائج مختبرات دلتا؟",
        "هل يوجد رسوم إضافية على نتائج حكيم لاب؟",
        "ما طريقة التواصل مع خدمة العملاء؟",
        "كيف أستفيد من خدمات المنصة؟",
        "هل تقبلون تقارير المستشفيات؟",
        "ما صيغة التقرير المقبولة؟",
        "هل يمكن تفسير نتيجة تحليل الدم؟",
        "ما حدود خدمة تفسير التحاليل؟",
        "هل تفسير التحليل يغني عن زيارة الطبيب؟",
        "هل يمكنني حجز استشارة من التطبيق؟",
        "كيف أطلب خدمة جديدة؟",
        "هل توجد زيارة منزلية لسحب الدم؟",
        "كم رسوم زيارة الممرضة المنزلية؟",
        "هل يمكن الاطلاع على تفاصيل الحجز؟",
        "ما طرق الدفع المتاحة؟",
        "هل تقبلون تابي وتمارا؟",
        "هل يمكن الدفع بالبطاقة؟",
        "كيف أتابع طلب الاسترداد؟",
        "هل يمكن إلغاء الاستشارة العاجلة؟",
        "ما شروط إلغاء الموعد العادي؟",
        "هل بياناتي الطبية محفوظة بسرية؟"
    ],

    "handoff": [
        "أريد تغيير موعد الاستشارة أونلاين.",
        "أحتاج تعديل موعد استشارتي.",
        "أريد نقل الاستشارة إلى يوم آخر.",
        "موعد الاستشارة لا يناسبني وأريد تغييره.",
        "أبغى أقدم موعد الاستشارة.",
        "أبغى أأخر موعد الاستشارة.",
        "أريد إعادة جدولة الاستشارة الطبية.",
        "ممكن تغيرون لي موعد الاستشارة؟",
        "أريد تعديل تاريخ الاستشارة المحجوزة.",
        "هل يمكن نقل موعد استشارتي؟",
        "أريد تغيير موعد زيارة الممرضة.",
        "أحتاج تعديل موعد سحب الدم المنزلي.",
        "موعد الزيارة المنزلية لا يناسبني.",
        "أريد نقل موعد سحب العينة إلى يوم آخر.",
        "أريد إعادة جدولة زيارة الممرضة.",
        "أبغى أغير وقت الزيارة المنزلية.",
        "أحتاج موعدًا جديدًا لسحب الدم بالمنزل.",
        "ممكن تعدلون موعد زيارة التمريض؟",
        "أريد تأجيل موعد سحب الدم المنزلي.",
        "أريد تقديم موعد الزيارة المنزلية."
    ],

    "collection_mixed": [
        "دفعت قيمة الحجز لكنه لا يظهر في التطبيق.",
        "تم خصم المبلغ ولم يتأكد الحجز.",
        "دفعت للاستشارة ولم يصلني تأكيد.",
        "المبلغ انخصم لكن الحجز غير موجود.",
        "أرسلت الدفع ولم يتم إنشاء الموعد.",
        "تم الدفع لكن لم يظهر لي الحجز.",
        "عندي عملية دفع ناجحة والحجز لم يكتمل.",
        "دفعت مرتين ولم يظهر موعدي.",
        "المبلغ انسحب من البطاقة ولم أحصل على الخدمة.",
        "أحتاج مساعدة لأن الدفع لم يكتمل بعدها الحجز.",
        "أريد طلب استرداد المبلغ.",
        "أرغب في استرجاع قيمة الحجز.",
        "أحتاج أرفع طلب استرداد دفعة.",
        "أريد استرجاع المبلغ الذي دفعته.",
        "كيف أقدم طلب استرداد للمبلغ المدفوع؟",
        "أبغى أسترجع قيمة الطلب.",
        "أريد إلغاء الطلب وطلب استرداد المبلغ.",
        "أحتاج مساعدة بخصوص استرجاع مبلغ الدفع.",
        "أرغب في استرداد رسوم الخدمة.",
        "أريد متابعة طلب استرداد مالي."
    ],

    "ambiguous": [
        "كم السعر؟",
        "أبغى أعرف التكلفة.",
        "كيف أقدر أستفيد؟",
        "عندي سؤال عن الخدمة.",
        "أبغى أطلبها.",
        "كم الرسوم؟",
        "متى الموعد؟",
        "أحتاج مساعدة في طلبي.",
        "هل هذا متاح؟",
        "أبغى أعرف التفاصيل."
    ]
}



# Balanced 170-question test distribution
LIMITS = {
    "reply": 50,
    "handoff": 20,
    "collection_mixed": 20,
    "ambiguous": 10,
}

CASES = {
    category: prompts[:LIMITS[category]]
    for category, prompts in CASES.items()
}

COLLECTION_HANDOFF_PROMPTS = {
    "دفعت قيمة الحجز لكنه لا يظهر في التطبيق.",
    "تم خصم المبلغ ولم يتأكد الحجز.",
    "دفعت للاستشارة ولم يصلني تأكيد.",
    "المبلغ انخصم لكن الحجز غير موجود.",
    "أرسلت الدفع ولم يتم إنشاء الموعد.",
    "تم الدفع لكن لم يظهر لي الحجز.",
    "عندي عملية دفع ناجحة والحجز لم يكتمل.",
    "دفعت مرتين ولم يظهر موعدي.",
    "المبلغ انسحب من البطاقة ولم أحصل على الخدمة.",
    "أحتاج مساعدة لأن الدفع لم يكتمل بعدها الحجز.",
    "أريد طلب استرداد المبلغ.",
    "أرغب في استرجاع قيمة الحجز.",
    "أحتاج أرفع طلب استرداد دفعة.",
    "أريد استرجاع المبلغ الذي دفعته.",
    "أبغى أسترجع قيمة الطلب.",
    "أريد إلغاء الطلب وطلب استرداد المبلغ.",
    "أحتاج مساعدة بخصوص استرجاع مبلغ الدفع.",
    "أرغب في استرداد رسوم الخدمة.",
    "أريد متابعة طلب استرداد مالي.",
}


def generate():
    questions = []
    counter = 1

    for category, prompts in CASES.items():
        for prompt in prompts:
            variants = [
                prompt,
                f"لو سمحت، {prompt}",
                f"{prompt} ممكن تساعدني؟",
                f"من فضلك، {prompt}",
                f"{prompt} أرجو المساعدة.",
            ]

            for variant in variants:
                questions.append({
                    "id": f"V9-{counter:03d}",
                    "category": category,
                    "prompt": variant,
                    "expected_action": (
                        (
                        "handoff"
                        if category == "handoff"
                        or prompt in COLLECTION_HANDOFF_PROMPTS
                        else "reply"
                    )
                    )
                })
                counter += 1

    if len(questions) != 500:
        raise ValueError(f"Expected 500 questions, got {len(questions)}")

    with open("intent_v9_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print("Generated:", len(questions))
    print("Distribution:", dict(Counter(q["category"] for q in questions)))
    return questions


def run(questions):
    if not TOKEN or TOKEN == "YOUR_CLIENT_TOKEN":
        raise SystemExit("Configure RAG_API_KEY first.")

    completed = set()
    try:
        with open(RESULTS, encoding="utf-8") as f:
            completed = {
                json.loads(line)["id"]
                for line in f if line.strip()
            }
    except FileNotFoundError:
        pass

    pending = [q for q in questions if q["id"] not in completed]
    print("Already completed:", len(completed))
    print("Remaining:", len(pending))

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    })

    with open(RESULTS, "a", encoding="utf-8") as output:
        for i, q in enumerate(pending, 1):
            result = {
                **q,
                "actual_action": "",
                "answer": "",
                "status": "",
                "latency": None,
                "error": ""
            }

            start = time.time()
            try:
                response = session.post(
                    API_URL,
                    json={"message": q["prompt"]},
                    timeout=180
                )
                result["latency"] = round(time.time() - start, 2)
                response.raise_for_status()
                data = response.json()

                result["actual_action"] = data.get("action", "")
                result["answer"] = data.get("answer", "")
                result["status"] = data.get("status", "UNKNOWN")

            except Exception as e:
                result["status"] = "REQUEST_FAILED"
                result["error"] = str(e)

            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()

            print(
                f"[{i}/{len(pending)}] {q['id']} | "
                f"{q['category']} | {result['actual_action']} | "
                f"{result['latency']}s"
            )

            if DELAY:
                time.sleep(DELAY)

    evaluate()


def evaluate():
    with open(RESULTS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    valid = [r for r in rows if r["status"] != "REQUEST_FAILED"]
    correct = [
        r for r in valid
        if r["expected_action"] == r["actual_action"]
    ]

    print("\n========== INTENT V2 REPORT ==========")
    print("Total:", len(rows))
    print("Correct:", len(correct))
    print(
        "Accuracy:",
        f"{len(correct)/len(valid)*100:.1f}%" if valid else "N/A"
    )

    for category in CASES:
        subset = [r for r in valid if r["category"] == category]
        print(
            f"{category}:",
            dict(Counter(r["actual_action"] for r in subset))
        )

    print("\n========== WRONG ACTIONS ==========")
    for r in valid:
        if r["expected_action"] != r["actual_action"]:
            print("\nID:", r["id"])
            print("Category:", r["category"])
            print("Question:", r["prompt"])
            print("Expected:", r["expected_action"])
            print("Actual:", r["actual_action"])
            print("Answer:", r["answer"])

    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "id", "category", "prompt", "expected_action",
            "actual_action", "status", "answer", "latency", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    latencies = [r["latency"] for r in valid if r["latency"] is not None]
    if latencies:
        print(f"\nAverage latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"Maximum latency: {max(latencies):.2f}s")

    print("Results:", RESULTS)
    print("CSV:", CSV_FILE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    questions = generate()

    if args.run:
        run(questions)
    else:
        print("Use --run to test.")
