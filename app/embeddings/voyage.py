import os
import time

import voyageai
from dotenv import load_dotenv

load_dotenv()

client = voyageai.Client(
    api_key=os.getenv("VOYAGE_API_KEY")
)


def create_embedding(text: str):
    result = client.embed(
        [text],
        model="voyage-4",
        input_type="query",
    )

    return result.embeddings[0]


def create_embeddings(
    texts: list[str],
    batch_size: int = 20,
):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        print(
            f"Embedding batch "
            f"{i + 1}-{min(i + batch_size, len(texts))} "
            f"of {len(texts)}"
        )

        result = client.embed(
            batch,
            model="voyage-4",
            input_type="document",
        )

        all_embeddings.extend(result.embeddings)

        if i + batch_size < len(texts):
            time.sleep(21)

    return all_embeddings