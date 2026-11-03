from app.documents.loader import load_document, prepare_documents
from app.embeddings.voyage import create_embeddings
from app.knowledge.repository import insert_knowledge_many


def ingest_document(file_path: str, client_id: int):
    document = load_document(file_path)

    return ingest_data(
        document,
        client_id,
    )


def ingest_data(document, client_id: int):
    prepared_documents = prepare_documents(document)

    embeddings = create_embeddings(prepared_documents)

    insert_knowledge_many(
        contents=prepared_documents,
        embeddings=embeddings,
        client_id=client_id,
    )

    return len(prepared_documents)
