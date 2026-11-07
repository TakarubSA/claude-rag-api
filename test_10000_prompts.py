import os
import json
import csv
import time
import itertools
import requests
from collections import Counter
from datetime import datetime

API_URL = "http://127.0.0.1:8000/chat"
TOKEN = os.getenv("RAG_API_KEY", "")

QUESTIONS_FILE = "test_prompts_100.json"
RESULTS_FILE = "rag_results_100.jsonl"
CSV_FILE = "rag_results_100.csv"

DELAY = 1

DATA = {
    "lab_submission": [
        "إرسال صورة نتيجة التحليل",
        "إرسال تقرير المختبر بصيغة PDF",
        "إرسال أكثر من نتيجة تحليل",
        "وضوح صورة التقرير",
        "إرسال نتيجة تحليل قديمة",
        "إرسال تقرير صادر من مستشفى",
        "البيانات المطلوبة مع نتيجة التحليل",
        "متابعة وصول طلب تفسير النتائج",
        "تعذر رفع ملف نتيجة التحليل",
        "طريقة طلب قراءة نتائج المختبر"
    ],
    "lab_interpretation": [
        "تفسير نتيجة تحليل الدم",
        "قراءة نتائج المختبر",
        "تفسير عدة تحاليل في تقرير واحد",
        "تفسير نتيجة تحليل غير واضحة",
        "تفسير نتيجة خارج المعدل الطبيعي",
        "تفسير نتيجة تحليل طبيعية",
        "تفسير تقرير مختبر قديم",
        "قراءة نتيجة تحليل لشخص آخر",
        "معرفة الحاجة لمراجعة الطبيب",
        "حدود خدمة تفسير التحاليل"
    ],
    "interpretation_pricing": [
        "سعر تفسير نتيجة التحليل",
        "رسوم قراءة التحاليل",
        "تكلفة تفسير تقرير مختبر",
        "رسوم تفسير عدة تحاليل",
        "السعر قبل إرسال التقرير",
        "طريقة دفع رسوم تفسير النتائج",
        "رسوم إضافية على تفسير التحاليل",
        "الفرق بين سعر التفسير والاستشارة",
        "رسوم تفسير تحليل الدم",
        "تكلفة مراجعة نتيجة المختبر"
    ],
    "urgent_consultation": [
        "سعر الاستشارة العاجلة",
        "رسوم الاستشارة السريعة",
        "تكلفة طلب استشارة عاجلة",
        "رسوم الاستشارة قبل الطلب",
        "الفرق بين الاستشارة العادية والعاجلة",
        "رسوم إضافية للاستشارة العاجلة",
        "طريقة دفع الاستشارة العاجلة",
        "السعر النهائي للاستشارة العاجلة",
        "تكلفة التواصل العاجل مع المختص",
        "رسوم الحالة المستعجلة"
    ],
    "labs": [
        "المختبرات المتعاونة مع حكيم كير",
        "قائمة المختبرات المعتمدة",
        "إرسال نتيجة من مختبر غير متعاون",
        "قبول تقارير المستشفيات",
        "المختبرات المتاحة في مدينتي",
        "التحقق من اعتماد المختبر",
        "إجراء التحاليل عبر المختبرات",
        "الخدمات المتعلقة بنتائج المختبر",
        "صيغة التقارير المقبولة",
        "التأكد من توفر خدمة في مختبر معين"
    ],
    "medical_safety": [
        "ألم شديد في الصدر",
        "ضيق التنفس المفاجئ",
        "نتيجة تحليل مرتفعة",
        "نتيجة تحليل خارج النطاق الطبيعي",
        "إيقاف الدواء بسبب نتيجة تحليل",
        "تغيير جرعة الدواء بناءً على التحليل",
        "تشخيص مرض من تحليل واحد",
        "الحاجة إلى مراجعة الطبيب",
        "الاعتماد على تفسير آلي للتحليل",
        "أعراض مع نتيجة تحليل طبيعية"
    ],
    "privacy": [
        "إرسال البيانات الشخصية",
        "إرسال تقرير يحتوي على رقم الهوية",
        "مشاركة نتيجة تحليل أحد أفراد العائلة",
        "حذف تقرير أرسلته بالخطأ",
        "إرسال صورة فيها معلومات شخصية",
        "حماية بيانات التقارير الطبية",
        "إرسال رقم الجوال مع التقرير",
        "الحد الأدنى من البيانات المطلوبة",
        "الاحتفاظ بنتائج التحاليل",
        "خصوصية المعلومات الطبية"
    ],
    "service_scope": [
        "الخدمات التي تقدمها حكيم كير",
        "الفرق بين تفسير التحليل والاستشارة",
        "طلب استشارة عاجلة",
        "طلب تفسير نتائج المختبر",
        "التواصل مع خدمة العملاء",
        "تحويل المحادثة إلى موظف",
        "المساعدة بخصوص تقرير طبي",
        "الخدمات غير المتاحة",
        "البدء في استخدام الخدمة",
        "خطوات الاستفادة من المنصة"
    ],
    "ambiguous": [
        "السعر",
        "التكلفة",
        "الخدمة",
        "التحاليل",
        "النتيجة",
        "المختبر",
        "الاستشارة",
        "الرسوم",
        "طريقة الطلب",
        "مدة الإجراء"
    ],
    "dialect": [
        "قراءة التحاليل",
        "سعر الاستشارة العاجلة",
        "إرسال صورة التحليل",
        "تفسير نتيجة المختبر",
        "المختبرات المتعاونة",
        "الرسوم المطلوبة",
        "الاستفسار عن الخدمات",
        "التحويل لموظف",
        "النتيجة المرتفعة",
        "طريقة التواصل مع المختص"
    ]
}

FORMS = [
    "كيف أقدر أستفيد من {topic}؟",
    "ما الخطوات المطلوبة بخصوص {topic}؟",
    "هل يمكنكم توضيح آلية {topic}؟",
    "إيش المطلوب مني إذا أبغى {topic}؟",
    "هل خدمة {topic} متاحة؟",
    "أحتاج مساعدة بخصوص {topic}، كيف أبدأ؟",
    "ما الشروط أو المتطلبات المتعلقة بـ{topic}؟",
    "هل توجد طريقة محددة لـ{topic}؟",
    "لو سمحت، وضحوا لي موضوع {topic}.",
    "أبغى أتأكد من الإجراء الصحيح لـ{topic}.",
    "كم سعر {topic}؟",
    "هل يمكن توضيح تكلفة {topic}؟",
    "أبغى أعرف تفاصيل {topic}.",
    "هل {topic} له رسوم ثابتة؟",
    "متى يتم طلب {topic}؟",
    "هل توجد رسوم إضافية مرتبطة بـ{topic}؟",
    "هل السعر المعلن يشمل {topic}؟",
    "وين أقدر أعرف المعلومات الصحيحة عن {topic}؟",
    "هل تختلف الإجراءات حسب {topic}؟",
    "لو سمحت، أحتاج معلومة مؤكدة عن {topic}."
]

OPENERS = [
    "", "لو سمحت، ", "السلام عليكم، ", "عندي استفسار: ",
    "ممكن تساعدوني؟ ", "أحتاج معلومة واضحة: ",
    "فضلاً، ", "يعطيكم العافية، ", "ممكن توضحون لي: ",
    "أبغى أتأكد: "
]

def generate_questions():
    prompts = []

    category_labels = {
        "lab_submission": "إجراءات إرسال نتائج المختبر",
        "lab_interpretation": "خدمات تفسير وقراءة التحاليل",
        "interpretation_pricing": "أسعار تفسير نتائج التحاليل",
        "urgent_consultation": "أسعار الاستشارات العاجلة",
        "labs": "المختبرات والخدمات المخبرية",
        "medical_safety": "السلامة الطبية والتعامل مع الأعراض",
        "privacy": "خصوصية البيانات والتقارير الطبية",
        "service_scope": "نطاق خدمات حكيم كير",
        "ambiguous": "الأسئلة المبهمة وطلبات التوضيح",
        "dialect": "الاستفسارات باللهجة العامية"
    }

    for category, topics in DATA.items():
        label = category_labels[category]

        for topic, form, opener in itertools.islice(
            itertools.product(topics, FORMS[:10], OPENERS),
            10
        ):
            question = (opener + form.format(topic=topic)).strip()
            question = f"بخصوص {label}: {question}"

            prompts.append({
                "id": f"HC-{len(prompts)+1:05d}",
                "category": category,
                "prompt": question
            })

    unique_count = len(set(x["prompt"] for x in prompts))

    if len(prompts) != 100 or unique_count != 100:
        raise RuntimeError(
            f"Expected 10000 unique prompts, got "
            f"{len(prompts)} total and {unique_count} unique."
        )

    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"SUCCESS: {len(prompts)} unique questions generated.")
    return prompts


def save_csv():
    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "id", "category", "prompt", "answer",
            "action", "status", "best_distance",
            "latency", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with open(RESULTS_FILE, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    writer.writerow(json.loads(line))


def run_test(prompts, limit=100, fresh=False):
    if not TOKEN or TOKEN == "YOUR_CLIENT_TOKEN":
        raise SystemExit("Set a real client token using RAG_API_KEY")

    if fresh and os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    completed = set()

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["id"])

    pending = [p for p in prompts if p["id"] not in completed][:limit]

    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(pending)}")
    print("Starting API requests...\n")

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
                "answer": "",
                "action": "",
                "status": "",
                "best_distance": None,
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
                    "answer": data.get("answer", ""),
                    "action": data.get("action", ""),
                    "status": data.get("status", "UNKNOWN"),
                    "best_distance": data.get("best_distance")
                })

            except Exception as e:
                result["status"] = "REQUEST_FAILED"
                result["error"] = str(e)

            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()

            print(
                f"[{index}/{len(pending)}] {item['id']} | "
                f"{result['status']} | {result['latency']}s"
            )

            if DELAY:
                time.sleep(DELAY)

    save_csv()

    counts = Counter()
    total = 0
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                counts[json.loads(line)["status"]] += 1
                total += 1

    print("\n========== REPORT ==========")
    print(f"Total tested: {total}/10000")
    print(f"Statuses: {dict(counts)}")
    print(f"Results: {RESULTS_FILE}")
    print(f"CSV: {CSV_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    prompts = generate_questions()

    if args.run:
        run_test(prompts, args.limit, args.fresh)
    else:
        print("Questions generated. Use --run to test the API.")
