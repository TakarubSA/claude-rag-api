from app.documents.loader import load_document, prepare_documents
from app.embeddings.voyage import create_embeddings
from app.knowledge.repository import insert_knowledge


def ingest_document(file_path: str, client_id: int):
    document = load_document(file_path)

    return ingest_data(
        document,
        client_id,
    )


def ingest_data(document, client_id: int):
    prepared_documents = prepare_documents(document)

    embeddings = create_embeddings(prepared_documents)

    for content, embedding in zip(prepared_documents, embeddings):
        insert_knowledge(
            content=content,
            embedding=embedding,
            client_id=client_id,
        )

    return len(prepared_documents)