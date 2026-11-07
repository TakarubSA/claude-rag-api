import json
import csv
import time
import requests
from collections import Counter
from pathlib import Path

from test_10000_prompts import API_URL, TOKEN, DELAY

QUESTIONS_FILE = "intent_questions_100.json"
RESULTS_FILE = "intent_results_100.jsonl"
CSV_FILE = "intent_results_100.csv"

CASES = {
    "reply": [
        ("lab_prices", "كم سعر تحليل الدم؟"),
        ("lab_prices", "أبغى أعرف أسعار التحاليل المتوفرة."),
        ("lab_prices", "هل عندكم قائمة بأسعار الباقات؟"),
        ("lab_prices", "وين أقدر أشوف سعر تحليل فيتامين د؟"),
        ("lab_prices", "هل أسعار التحاليل موجودة في التطبيق؟"),
        ("lab_prices", "كم تكلفة باقة التحاليل الشاملة؟"),
        ("lab_prices", "أحتاج أعرف سعر تحليل الغدة الدرقية."),
        ("lab_prices", "هل تختلف أسعار التحاليل من مختبر لآخر؟"),
        ("lab_prices", "كيف أعرف أسعار الباقات الطبية؟"),
        ("lab_prices", "أين أجد أسعار خدمات المختبرات؟"),

        ("interpretation_price", "كم سعر تفسير نتائج التحاليل؟"),
        ("interpretation_price", "كم رسوم قراءة نتيجة التحليل؟"),
        ("interpretation_price", "هل تفسير التحاليل مقابل 15 ريال؟"),
        ("interpretation_price", "كم تكلفة تفسير تقرير المختبر؟"),
        ("interpretation_price", "هل توجد رسوم على قراءة النتائج؟"),
        ("interpretation_price", "كم سعر تفسير نتائج مختبرات دلتا؟"),
        ("interpretation_price", "هل تفسير نتائج المختبرات الأخرى برسوم؟"),
        ("interpretation_price", "هل رسوم تفسير التحاليل ثابتة؟"),
        ("interpretation_price", "هل سعر التفسير يختلف حسب عدد التحاليل؟"),
        ("interpretation_price", "هل توجد رسوم إضافية لقراءة النتائج؟"),

        ("hakeem_lab", "حللت عن طريق حكيم لاب، هل التفسير مجاني؟"),
        ("hakeem_lab", "وصلني رابط من حكيم لاب، ماذا أفعل؟"),
        ("hakeem_lab", "هل أدفع رسومًا إضافية لتفسير نتائج حكيم لاب؟"),
        ("hakeem_lab", "كيف أحصل على تفسير نتائجي من رابط حكيم لاب؟"),
        ("hakeem_lab", "هل الطبيب يتواصل معي من خلال رابط حكيم لاب؟"),
        ("hakeem_lab", "هل أحتاج أحجز خدمة تفسير إذا عندي رابط حكيم لاب؟"),
        ("hakeem_lab", "هل يوجد مبلغ إضافي لمراجعة نتائج حكيم لاب؟"),
        ("hakeem_lab", "عندي نتائج من حكيم لاب، كيف تتم مراجعتها؟"),
        ("hakeem_lab", "هل الرابط المرسل من حكيم لاب يكفي لمراجعة النتائج؟"),
        ("hakeem_lab", "هل خدمة تفسير نتائج حكيم لاب عليها رسوم؟"),

        ("refund_info", "متى يرجع المبلغ بعد الموافقة على الاسترداد؟"),
        ("refund_info", "كم يستغرق وصول المبلغ المسترد؟"),
        ("refund_info", "هل يرجع المبلغ خلال يوم عمل؟"),
        ("refund_info", "هل الاسترداد يشمل الدفع بالبطاقة؟"),
        ("refund_info", "هل يمكن استرداد مبلغ الدفع إلى المحفظة؟"),
        ("refund_info", "متى يظهر المبلغ المسترد في حسابي؟"),
        ("refund_info", "هل تختلف مدة الاسترداد حسب طريقة الدفع؟"),
        ("refund_info", "كم المدة المتوقعة لإرجاع المبلغ؟"),
        ("refund_info", "هل يتم استرداد المبلغ خلال يوم عمل واحد؟"),
        ("refund_info", "كيف أعرف سياسة استرداد المبالغ؟"),

        ("service_info", "ما الخدمات التي تقدمها حكيم كير؟"),
        ("service_info", "هل تقدمون خدمة تفسير نتائج التحاليل؟"),
        ("service_info", "ما الفرق بين تفسير التحليل والاستشارة الطبية؟"),
        ("service_info", "هل يمكن تفسير نتائج من مختبر آخر؟"),
        ("service_info", "هل يمكن إرسال تقرير المختبر بصيغة PDF؟"),
        ("service_info", "ما الجهات المتعاونة مع حكيم كير؟"),
        ("service_info", "هل حكيم كير مختبر طبي؟"),
        ("service_info", "هل يمكنني الاطلاع على الخدمات من التطبيق؟"),
        ("service_info", "كيف أبدأ باستخدام خدمات حكيم كير؟"),
        ("service_info", "هل خدمة تفسير النتائج متاحة؟"),
    ],

    "handoff": [
        ("reschedule_online", "أريد تغيير موعد الاستشارة أونلاين."),
        ("reschedule_online", "هل يمكنكم تعديل موعد استشارتي؟"),
        ("reschedule_online", "أحتاج أحجز الاستشارة في وقت آخر بدل الموعد الحالي."),
        ("reschedule_online", "موعد استشارتي لا يناسبني وأريد تغييره."),
        ("reschedule_online", "أبغى أقدم موعد الاستشارة."),
        ("reschedule_online", "أبغى أأخر موعد الاستشارة."),
        ("reschedule_online", "هل يمكن نقل استشارتي إلى يوم آخر؟"),
        ("reschedule_online", "أحتاج إعادة جدولة الاستشارة الطبية."),
        ("reschedule_online", "أريد تعديل تاريخ الاستشارة المحجوزة."),
        ("reschedule_online", "ممكن تغيرون لي موعد الاستشارة؟"),

        ("reschedule_nurse", "أريد تغيير موعد زيارة الممرضة المنزلية."),
        ("reschedule_nurse", "أحتاج تعديل موعد سحب الدم من المنزل."),
        ("reschedule_nurse", "موعد الزيارة المنزلية لا يناسبني."),
        ("reschedule_nurse", "أريد نقل موعد سحب العينة إلى يوم آخر."),
        ("reschedule_nurse", "هل يمكن إعادة جدولة زيارة الممرضة؟"),
        ("reschedule_nurse", "أبغى أغير وقت الزيارة المنزلية."),
        ("reschedule_nurse", "أحتاج موعدًا جديدًا لسحب الدم بالمنزل."),
        ("reschedule_nurse", "ممكن تعدلون موعد زيارة التمريض؟"),
        ("reschedule_nurse", "أريد تأجيل موعد سحب الدم المنزلي."),
        ("reschedule_nurse", "أريد تقديم موعد الزيارة المنزلية."),

        ("payment_booking", "دفعت قيمة الحجز لكنه لا يظهر في التطبيق."),
        ("payment_booking", "تم خصم المبلغ ولم يتأكد الحجز."),
        ("payment_booking", "دفعت للاستشارة ولم يصلني تأكيد."),
        ("payment_booking", "المبلغ انخصم لكن الحجز غير موجود."),
        ("payment_booking", "أرسلت الدفع ولم يتم إنشاء الموعد."),
        ("payment_booking", "تم الدفع لكن لم يظهر لي الحجز."),
        ("payment_booking", "عندي عملية دفع ناجحة والحجز لم يكتمل."),
        ("payment_booking", "دفعت مرتين ولم يظهر موعدي."),
        ("payment_booking", "المبلغ انسحب من البطاقة ولم أحصل على الخدمة."),
        ("payment_booking", "أحتاج حل مشكلة دفع لم يكتمل بعدها الحجز."),

        ("refund_request", "أريد طلب استرداد المبلغ."),
        ("refund_request", "أرغب في استرجاع قيمة الحجز."),
        ("refund_request", "أحتاج أرفع طلب استرداد دفعة."),
        ("refund_request", "أريد استرجاع المبلغ الذي دفعته."),
        ("refund_request", "كيف أقدم طلب استرداد للمبلغ المدفوع؟"),
        ("refund_request", "أبغى أسترجع قيمة الطلب."),
        ("refund_request", "أريد إلغاء الطلب وطلب استرداد المبلغ."),
        ("refund_request", "أحتاج مساعدة بخصوص استرجاع مبلغ الدفع."),
        ("refund_request", "أرغب في استرداد رسوم الخدمة."),
        ("refund_request", "أريد متابعة طلب استرداد مالي."),
    ],

    "clarification": [
        ("ambiguous", "كم السعر؟"),
        ("ambiguous", "أبغى أعرف التكلفة."),
        ("ambiguous", "كيف أقدر أستفيد؟"),
        ("ambiguous", "عندي سؤال عن الخدمة."),
        ("ambiguous", "أبغى أطلبها."),
        ("ambiguous", "كم الرسوم؟"),
        ("ambiguous", "متى الموعد؟"),
        ("ambiguous", "أحتاج مساعدة في طلبي."),
        ("ambiguous", "هل هذا متاح؟"),
        ("ambiguous", "أبغى أعرف التفاصيل."),
    ]
}


def generate_questions():
    prompts = []
    counter = 1

    for expected, cases in CASES.items():
        for category, prompt in cases:
            prompts.append({
                "id": f"INT-{counter:03d}",
                "category": category,
                "expected_action": "reply" if expected == "clarification" else expected,
                "expected_type": expected,
                "prompt": prompt
            })
            counter += 1

    if len(prompts) != 100:
        raise RuntimeError(f"Expected 100 questions, got {len(prompts)}")

    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print("Generated 100 intent test cases.")
    print("Expected distribution:", dict(Counter(x["expected_type"] for x in prompts)))
    return prompts


def run_test(prompts):
    if not TOKEN or TOKEN == "YOUR_CLIENT_TOKEN":
        raise SystemExit("ERROR: Configure RAG_API_KEY first.")

    completed = set()

    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["id"])

    pending = [p for p in prompts if p["id"] not in completed]

    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(pending)}")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    })

    with open(RESULTS_FILE, "a", encoding="utf-8") as output:
        for index, item in enumerate(pending, 1):
            start = time.time()

            result = {
                **item,
                "actual_action": "",
                "answer": "",
                "status": "",
                "latency": None,
                "error": ""
            }

            try:
                response = session.post(
                    API_URL,
                    json={"message": item["prompt"]},
                    timeout=180
                )
                result["latency"] = round(time.time() - start, 2)
                response.raise_for_status()
                data = response.json()

                result.update({
                    "actual_action": data.get("action", ""),
                    "answer": data.get("answer", ""),
                    "status": data.get("status", "UNKNOWN")
                })

            except Exception as e:
                result["status"] = "REQUEST_FAILED"
                result["error"] = str(e)

            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()

            print(
                f"[{index}/{len(pending)}] {item['id']} | "
                f"Expected: {item['expected_type']} | "
                f"Actual: {result['actual_action']} | "
                f"{result['latency']}s"
            )

            if DELAY:
                time.sleep(DELAY)

    evaluate()


def evaluate():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = {x["id"]: x for x in json.load(f)}

    results = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    rows = []
    for result in results:
        expected = result["expected_action"]
        actual = result["actual_action"]

        row = {
            **result,
            "correct_action": expected == actual if actual else False
        }
        rows.append(row)

    completed = [r for r in rows if r["status"] != "REQUEST_FAILED"]
    correct = sum(r["correct_action"] for r in completed)

    print("\n========== INTENT ACCURACY REPORT ==========")
    print(f"Completed: {len(completed)}/100")
    print(f"Request failures: {len(rows) - len(completed)}")
    print(f"Correct actions: {correct}/{len(completed)}")
    print(
        f"Intent accuracy: "
        f"{(correct / len(completed) * 100):.1f}%"
        if completed else "Intent accuracy: N/A"
    )

    print("\n========== EXPECTED vs ACTUAL ==========")
    for expected_type in ("reply", "handoff", "clarification"):
        subset = [r for r in rows if r["expected_type"] == expected_type]
        counts = Counter(r["actual_action"] or "FAILED" for r in subset)
        print(f"{expected_type}: {dict(counts)}")

    print("\n========== WRONG ACTIONS ==========")
    wrong = [r for r in rows if not r["correct_action"]]
    for r in wrong:
        print(f"\nID: {r['id']}")
        print(f"Type: {r['expected_type']}")
        print(f"Question: {r['prompt']}")
        print(f"Expected: {r['expected_action']}")
        print(f"Actual: {r['actual_action']}")
        print(f"Answer: {r['answer']}")

    latencies = [r["latency"] for r in completed if r["latency"] is not None]
    if latencies:
        print("\n========== PERFORMANCE ==========")
        print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"Maximum latency: {max(latencies):.2f}s")

    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "id", "category", "expected_type", "prompt",
            "expected_action", "actual_action", "correct_action",
            "answer", "status", "latency", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults: {RESULTS_FILE}")
    print(f"CSV: {CSV_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    prompts = generate_questions()

    if args.run:
        run_test(prompts)
    else:
        print("Use --run to test the API.")
