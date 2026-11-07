import json
import os

from app.documents.loader import load_document, prepare_documents, split_text
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


SUPPORTED_EXTENSIONS = {".json", ".md", ".txt"}


def ingest_file(filename: str, content: str, client_id: int) -> int:
    """Ingest an uploaded .json, .md or .txt file. Returns the entry count."""
    extension = os.path.splitext(filename.lower())[1]

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only .json, .md and .txt files are supported")

    if extension == ".json":
        try:
            entries = prepare_documents(json.loads(content))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
    else:
        entries = split_text(content, markdown=extension == ".md")

    if not entries:
        raise ValueError("No knowledge entries found in the file")

    insert_knowledge_many(
        contents=entries,
        embeddings=create_embeddings(entries),
        client_id=client_id,
    )

    return len(entries)
