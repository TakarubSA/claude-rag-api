from app.services.rag import answer_question

question = "أبغى أسترد المبلغ"

answer = answer_question(question)

print("Question:", question)
print("Answer:", answer)