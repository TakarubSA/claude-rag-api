import json
import re


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

MAX_CHUNK_CHARS = 1200


def split_text(content: str, markdown: bool = False) -> list[str]:
    """
    Split a .txt / .md file into knowledge entries.

    Markdown is split per heading section; each entry keeps its heading so
    it still makes sense when retrieved alone. Paragraphs are then packed
    into entries of up to MAX_CHUNK_CHARS.
    """
    content = content.replace("\r\n", "\n")

    if markdown:
        sections = re.split(r"(?m)^(?=#{1,6}\s)", content)
    else:
        sections = [content]

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        heading = ""

        if markdown and section.startswith("#"):
            heading, _, section = section.partition("\n")
            heading = heading.strip()

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]

        if not paragraphs:
            # A heading with no body is not useful knowledge on its own.
            continue

        current = ""

        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARS:
                chunks.append(_with_heading(heading, current))
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph

        chunks.append(_with_heading(heading, current))

    return chunks


def _with_heading(heading: str, body: str) -> str:
    return f"{heading}\n{body}" if heading else body
