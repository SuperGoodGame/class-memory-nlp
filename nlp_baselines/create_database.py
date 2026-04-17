from __future__ import annotations

import argparse
import os
import shutil

from .data_utils import load_text, split_text
from .embeddings_utils import DEFAULT_EMBEDDING_MODEL, get_embeddings
from .vector_store import LocalVectorStore


CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def load_documents() -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    for filename in sorted(os.listdir(DATA_PATH)):
        if filename.endswith(".md"):
            path = os.path.join(DATA_PATH, filename)
            documents.append((path, load_text(path)))
    return documents


def save_to_chroma(
    texts: list[str],
    metadatas: list[dict[str, object]],
    *,
    persist_directory: str = CHROMA_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    store = LocalVectorStore.build(
        texts=texts,
        metadatas=metadatas,
        embedding_model=embedding_model,
    )
    store.save(persist_directory)
    print(f"Saved {store.count()} chunks to {persist_directory}.")


def generate_data_store(
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_directory: str = CHROMA_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    documents = load_documents()
    chunks = []
    for source_path, text in documents:
        chunks.extend(
            split_text(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_path=source_path,
            )
        )
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    save_to_chroma(
        [chunk.page_content for chunk in chunks],
        [chunk.metadata for chunk in chunks],
        persist_directory=persist_directory,
        embedding_model=embedding_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Chroma database for Alice in Wonderland.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--persist-directory", type=str, default=CHROMA_PATH)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    generate_data_store(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_directory=args.persist_directory,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
