import os

from app.services.rag import answer_question

# Knowledge is per client: set TEST_CLIENT_ID to the client to query.
client_id = int(os.getenv("TEST_CLIENT_ID", "1"))

question = "أبغى أسترد المبلغ"

answer = answer_question(question, client_id=client_id)

print("Question:", question)
print("Answer:", answer)
