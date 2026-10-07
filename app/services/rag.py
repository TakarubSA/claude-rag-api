from app.embeddings.voyage import create_embedding
from app.knowledge.repository import search_knowledge
from app.llm.claude import (
    generate_answer,
    check_context_relevance,
    classify_knowledge_gap,
)


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

    # 4. Check if the retrieved context can actually answer the question
    relevance = check_context_relevance(
        question=question,
        context=context,
    )

    print("CONTEXT RELEVANCE:", relevance)
    answer = generate_answer(
            question,
            context,
        )

    # 5. If the context is enough, generate the answer
    if relevance == "ANSWERABLE":
        # answer = generate_answer(
        #     question,
        #     context,
        # )

        best_distance = results[0].distance if results else None

        return {
            "answer": answer,
            "best_distance": best_distance,
            "status": "KNOWN",
        }

    # 6. The context was not enough to answer the question
    gap_type = classify_knowledge_gap(
        question=question,
        context=context,
    )

    print("KNOWLEDGE GAP:", gap_type)

    return {
        "answer":answer,
        "best_distance": results[0].distance if results else None,
        "status": gap_type,
    }