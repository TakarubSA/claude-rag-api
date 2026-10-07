import json


def load_document(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_documents(document):
    prepared_documents = []

    items = extract_items(document)

    for item in items:
        text = extract_text(item)

        if text:
            prepared_documents.append(text)

    return prepared_documents


def extract_items(document):
    if isinstance(document, list):
        return document

    if isinstance(document, dict):
        for key in ["entries", "items", "data", "documents", "knowledge"]:
            value = document.get(key)

            if isinstance(value, list):
                return value

        return [document]

    return []


def extract_text(item):
    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return None

    # Informative knowledge
    for key in ["content", "text", "description", "information"]:
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    # Question / Answer
    question = item.get("question")
    answer = item.get("answer")

    if question and answer:
        return f"""
Question: {question}
Answer: {answer}
""".strip()

    return None