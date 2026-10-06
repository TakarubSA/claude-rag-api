import json


def load_document(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        document = json.load(file)

    return document


def prepare_documents(document):
    prepared_documents = []

    for item in document:
        text = f"""
Question: {item["question"]}
Answer: {item["answer"]}
""".strip()

        prepared_documents.append(text)

    return prepared_documents