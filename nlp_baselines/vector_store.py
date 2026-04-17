from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .embeddings_utils import DEFAULT_EMBEDDING_MODEL, get_embeddings


RECORDS_FILENAME = "records.json"
EMBEDDINGS_FILENAME = "embeddings.npy"


@dataclass
class RetrievedRecord:
    page_content: str
    metadata: dict[str, object]
    score: float


class LocalVectorStore:
    def __init__(self, records: list[dict[str, object]], embeddings: np.ndarray) -> None:
        self.records = records
        self.embeddings = embeddings.astype("float32")

    @classmethod
    def build(
        cls,
        *,
        texts: list[str],
        metadatas: list[dict[str, object]],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "LocalVectorStore":
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")
        embedder = get_embeddings(embedding_model)
        embeddings = np.asarray(embedder.embed_documents(texts), dtype="float32")
        embeddings = _normalize(embeddings)
        records = [
            {"page_content": text, "metadata": metadata}
            for text, metadata in zip(texts, metadatas, strict=True)
        ]
        return cls(records=records, embeddings=embeddings)

    @classmethod
    def load(cls, persist_directory: str) -> "LocalVectorStore":
        records_path = os.path.join(persist_directory, RECORDS_FILENAME)
        embeddings_path = os.path.join(persist_directory, EMBEDDINGS_FILENAME)
        with open(records_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        embeddings = np.load(embeddings_path)
        return cls(records=records, embeddings=embeddings)

    def save(self, persist_directory: str) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        records_path = os.path.join(persist_directory, RECORDS_FILENAME)
        embeddings_path = os.path.join(persist_directory, EMBEDDINGS_FILENAME)
        with open(records_path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, ensure_ascii=False, indent=2)
        np.save(embeddings_path, self.embeddings)

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[RetrievedRecord]:
        if not self.records:
            return []
        embedder = get_embeddings(embedding_model)
        query_embedding = np.asarray([embedder.embed_query(query)], dtype="float32")
        query_embedding = _normalize(query_embedding)
        scores = np.matmul(self.embeddings, query_embedding[0])
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievedRecord(
                page_content=str(self.records[index]["page_content"]),
                metadata=dict(self.records[index].get("metadata", {})),
                score=float(scores[index]),
            )
            for index in ranked_indices.tolist()
        ]

    def count(self) -> int:
        return len(self.records)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)
