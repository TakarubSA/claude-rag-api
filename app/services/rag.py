from app.embeddings.voyage import create_embedding
from app.knowledge.repository import search_knowledge
from app.llm.claude import (
    rewrite_query,
    generate_answer,
    check_context_relevance,
    classify_knowledge_gap,
)


def answer_question(question: str, client_id: int):
    # 1. Rewrite the user's question for better retrieval
    search_query = rewrite_query(question)

    print("ORIGINAL QUERY:", question)
    print("SEARCH QUERY:", search_query)

    # 2. Convert the rewritten query into an embedding
    query_embedding = create_embedding(search_query)

    # 3. Find relevant knowledge for this client only
    results = search_knowledge(
        query_embedding,
        client_id=client_id,
        limit=8,
    )

    # Debug: show retrieved knowledge
    for result in results:
        print("ID:", result.id)
        print("DISTANCE:", result.distance)
        print("CONTENT:", result.content)
        print("---")

    # 4. Build the context that Claude will receive
    context = "\n\n".join(
        result.content
        for result in results
    )

    # Debug: show complete context
    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)
    print("========================================\n")

    # 5. Check if the retrieved context can answer the original question
    relevance = check_context_relevance(
        question=question,
        context=context,
    )

    print("CONTEXT RELEVANCE:", relevance)

    # 6. If the context is enough, generate the answer
    if relevance == "ANSWERABLE":
        answer = generate_answer(
            question=question,
            context=context,
        )

        best_distance = results[0].distance if results else None

        return {
            "answer": answer,
            "best_distance": best_distance,
            "status": "KNOWN",
        }

    # 7. Context was not enough to answer the question
    gap_type = classify_knowledge_gap(
        question=question,
        context=context,
    )

    print("KNOWLEDGE GAP:", gap_type)

    best_distance = results[0].distance if results else None

    return {
        "answer": "عذرًا، لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال حاليًا.",
        "best_distance": best_distance,
        "status": gap_type,
    }