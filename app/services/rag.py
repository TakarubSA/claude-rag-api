from app.embeddings.voyage import create_embedding
from app.knowledge.repository import search_knowledge
from app.llm.claude import generate_answer


def answer_question(question: str, client_id: int):
    # 1. Convert the user's question into an embedding
    query_embedding = create_embedding(question)

    # 2. Find relevant knowledge for this client only
    results = search_knowledge(
        query_embedding,
        client_id=client_id,
        limit=3,
    )


    for result in results:
            print("ID:", result.id)
            print("DISTANCE:", result.distance)
            print("CONTENT:", result.content)
            print("---")



    # 3. Build the context that Claude will receive
    context = "\n\n".join(
        result.content
        for result in results
    )

    # 4. Ask Claude to answer using the retrieved context
    answer = generate_answer(
        question,
        context,
    )

    return answer