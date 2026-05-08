from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .benchmark_loaders import SUPPORTED_LONGBENCH_DATASETS, BenchmarkExample, load_longbench_sample
from .data_utils import split_text
from .embeddings_utils import DEFAULT_EMBEDDING_MODEL
from .local_generation import local_chat_completion
from .summary_utils import format_summary_record, parse_summary_output
from .vector_store import LocalVectorStore


STRUCTURED_MEMORY_DIR = Path("data/longbench/structured_memory")
DEFAULT_SECTION_SIZE = 10000
DEFAULT_SECTION_OVERLAP = 500


def safe_example_id(example_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", example_id)


def memory_cache_path(dataset: str, example_id: str) -> Path:
    return STRUCTURED_MEMORY_DIR / dataset / f"{safe_example_id(example_id)}.jsonl"


def build_summary_prompt(section_text: str) -> str:
    return (
        "Summarize the following long-document section into structured memory for QA.\n"
        "Return valid JSON only with these keys:\n"
        "{\n"
        '  "section_summary": "2-4 sentence concise summary",\n'
        '  "key_entities": ["important people, systems, datasets, places"],\n'
        '  "key_events": ["important actions, claims, findings, plot events"],\n'
        '  "exact_facts": ["specific facts with names, numbers, dates, labels, outcomes"],\n'
        '  "supporting_quotes": ["short exact quotes or phrases useful as evidence"]\n'
        "}\n\n"
        "Rules:\n"
        "- Preserve exact answerable facts, names, numbers, and causal relations.\n"
        "- Do not invent facts.\n"
        "- Keep each list compact; at most 8 items per list.\n"
        "- JSON only, no markdown.\n\n"
        f"Section:\n{section_text}"
    )


def summarize_section(section_text: str, *, timeout: int = 300) -> dict[str, object]:
    response = local_chat_completion(
        [
            {
                "role": "system",
                "content": "You build structured long-term memory records for long-context QA. Return JSON only.",
            },
            {"role": "user", "content": build_summary_prompt(section_text[:10000])},
        ],
        max_tokens=600,
        temperature=0.0,
        timeout=timeout,
        disable_thinking=True,
    )
    parsed = parse_summary_output(response.text)
    if not parsed.get("section_summary") and not parsed.get("exact_facts"):
        parsed["section_summary"] = section_text[:500].strip()
        parsed["supporting_quotes"] = [section_text[:180].strip()]
    parsed["_generation_latency"] = response.latency
    parsed["_prompt_tokens"] = response.prompt_tokens or 0
    parsed["_completion_tokens"] = response.completion_tokens or 0
    parsed["_finish_reason"] = response.finish_reason
    return parsed


def section_records(example: BenchmarkExample, *, section_size: int, section_overlap: int) -> list[dict[str, Any]]:
    chunks = split_text(
        example.context,
        chunk_size=section_size,
        chunk_overlap=section_overlap,
        source_path=f"longbench/{example.dataset}/{example.example_id}",
    )
    records: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        records.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "section_id": f"section_{index}",
                "raw_text": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )
    return records


def load_memory_records(dataset: str, example_id: str) -> list[dict[str, Any]]:
    path = memory_cache_path(dataset, example_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_memory_records(dataset: str, example_id: str, records: list[dict[str, Any]]) -> None:
    path = memory_cache_path(dataset, example_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["section_id"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def ensure_example_memory(
    example: BenchmarkExample,
    *,
    section_size: int = DEFAULT_SECTION_SIZE,
    section_overlap: int = DEFAULT_SECTION_OVERLAP,
    max_workers: int = 1,
    timeout: int = 300,
) -> list[dict[str, Any]]:
    sections = section_records(example, section_size=section_size, section_overlap=section_overlap)
    cached = load_memory_records(example.dataset, example.example_id)
    cached_by_id = {str(record["section_id"]): record for record in cached}
    missing = [section for section in sections if section["section_id"] not in cached_by_id]
    if not missing:
        return [cached_by_id[section["section_id"]] for section in sections]

    print(
        f"  Building structured memory for {example.dataset}/{example.example_id}: "
        f"{len(missing)}/{len(sections)} sections missing"
    )
    generated: list[dict[str, Any]] = []

    def build_one(section: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        summary = summarize_section(str(section["raw_text"]), timeout=timeout)
        formatted_text = format_summary_record(summary)
        return {
            **section,
            "summary": summary,
            "formatted_text": formatted_text,
            "source_char_count": len(str(section["raw_text"])),
            "summary_char_count": len(formatted_text),
            "build_latency": time.time() - started,
        }

    if max_workers <= 1:
        for section in missing:
            generated.append(build_one(section))
            cached_by_id[generated[-1]["section_id"]] = generated[-1]
            save_memory_records(example.dataset, example.example_id, list(cached_by_id.values()))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(build_one, section): section for section in missing}
            for future in as_completed(futures):
                record = future.result()
                generated.append(record)
                cached_by_id[record["section_id"]] = record
                save_memory_records(example.dataset, example.example_id, list(cached_by_id.values()))

    return [cached_by_id[section["section_id"]] for section in sections]


def build_memory_store(records: list[dict[str, Any]]) -> LocalVectorStore:
    return LocalVectorStore.build(
        texts=[str(record["formatted_text"]) for record in records],
        metadatas=[
            {
                "section_id": str(record["section_id"]),
                "example_id": str(record["example_id"]),
                "dataset": str(record["dataset"]),
            }
            for record in records
        ],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )


def retrieve_structured_memory(
    example: BenchmarkExample,
    *,
    summary_k: int,
    raw_k: int,
    section_size: int = DEFAULT_SECTION_SIZE,
    section_overlap: int = DEFAULT_SECTION_OVERLAP,
    max_workers: int = 1,
) -> tuple[list[str], float]:
    start = time.time()
    records = ensure_example_memory(
        example,
        section_size=section_size,
        section_overlap=section_overlap,
        max_workers=max_workers,
    )
    memory_store = build_memory_store(records)
    memory_docs = memory_store.similarity_search(example.question, k=summary_k, embedding_model=DEFAULT_EMBEDDING_MODEL)
    records_by_id = {str(record["section_id"]): record for record in records}
    memory_texts = [doc.page_content for doc in memory_docs]
    raw_candidates = [str(records_by_id[str(doc.metadata["section_id"])]["raw_text"]) for doc in memory_docs]

    raw_chunks = split_text(
        example.context,
        chunk_size=1200,
        chunk_overlap=200,
        source_path=f"longbench/{example.dataset}/{example.example_id}",
    )
    raw_store = LocalVectorStore.build(
        texts=[chunk.page_content for chunk in raw_chunks],
        metadatas=[chunk.metadata for chunk in raw_chunks],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    raw_docs = raw_store.similarity_search(example.question, k=raw_k, embedding_model=DEFAULT_EMBEDDING_MODEL)
    raw_candidates.extend(doc.page_content for doc in raw_docs)

    deduped: list[str] = []
    seen: set[str] = set()
    for text in raw_candidates:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return memory_texts + deduped[:raw_k], time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM-generated structured memory for LongBench samples.")
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_LONGBENCH_DATASETS))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--section-size", type=int, default=DEFAULT_SECTION_SIZE)
    parser.add_argument("--section-overlap", type=int, default=DEFAULT_SECTION_OVERLAP)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    for dataset in args.datasets:
        examples = load_longbench_sample(dataset, limit=args.limit)
        for index, example in enumerate(examples, 1):
            print(f"[memory] {dataset} {index}/{len(examples)} {example.example_id}")
            ensure_example_memory(
                example,
                section_size=args.section_size,
                section_overlap=args.section_overlap,
                max_workers=args.max_workers,
                timeout=args.timeout,
            )


if __name__ == "__main__":
    main()
