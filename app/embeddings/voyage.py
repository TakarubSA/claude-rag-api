from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-large"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def create_embedding(text: str):
    embeddings = create_embeddings(
        [text],
        input_type="query",
    )
    return embeddings[0]


def create_embeddings(
    texts: list[str],
    batch_size: int = 20,
    input_type: str = "document",
):
    if not texts:
        return []

    prefix = "query: " if input_type == "query" else "passage: "

    prepared_texts = [
        prefix + text.strip()
        for text in texts
    ]

    model = get_model()

    embeddings = model.encode(
        prepared_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()