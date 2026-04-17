from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    page_content: str
    metadata: dict[str, object]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    chapter_start = text.find("CHAPTER I.")
    if chapter_start != -1:
        text = text[chapter_start:]

    ending_marker = text.rfind("THE END")
    if ending_marker != -1:
        text = text[: ending_marker + len("THE END")]

    return text


def split_text(
    text: str,
    *,
    chunk_size: int = 300,
    chunk_overlap: int = 100,
    source_path: str | None = None,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": f"chunk_{chunk_index}",
                        "start_index": start,
                        "end_index": end,
                        "source_path": source_path or "",
                    },
                )
            )
            chunk_index += 1
        if end >= len(text):
            break
        start += step

    return chunks
