from app.llm.claude import generate_answer


question = "أبغى أسترد المبلغ"

context = """
Question: كيف أرجع فلوسي؟
Answer: يمكنك طلب استرجاع المبلغ خلال 14 يومًا من تاريخ استلام الطلب،
بشرط أن يكون المنتج بحالته الأصلية.
"""

answer = generate_answer(question, context)

print(answer)