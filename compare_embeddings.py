from __future__ import annotations

from embeddings_utils import cosine_similarity, get_embeddings


def main() -> None:
    embedding_function = get_embeddings()

    apple_vector = embedding_function.embed_query("apple")
    iphone_vector = embedding_function.embed_query("iphone")

    print(f"Vector length: {len(apple_vector)}")
    print(f"Cosine similarity (apple, iphone): {cosine_similarity(apple_vector, iphone_vector):.4f}")


if __name__ == "__main__":
    main()
