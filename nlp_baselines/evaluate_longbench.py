from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from .benchmark_loaders import SUPPORTED_LONGBENCH_DATASETS, BenchmarkExample, load_longbench_sample
from .data_utils import split_text
from .embeddings_utils import DEFAULT_EMBEDDING_MODEL
from .local_generation import local_chat_completion, local_model_name
from .longbench_structured_memory import retrieve_structured_memory
from .summary_utils import keyword_overlap_score
from .vector_store import LocalVectorStore


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
RETRIEVAL_JSON_PATH = RESULTS_DIR / "longbench_retrieval_only.json"
GENERATION_JSON_PATH = RESULTS_DIR / "local_generation_gemma4_longbench.json"
RETRIEVAL_CSV_PATH = TABLES_DIR / "longbench_retrieval_summary.csv"
GENERATION_CSV_PATH = TABLES_DIR / "longbench_generation_summary.csv"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens or not answer_tokens:
        return 0.0
    common = set(pred_tokens) & set(answer_tokens)
    if not common:
        return 0.0
    precision = sum(min(pred_tokens.count(token), answer_tokens.count(token)) for token in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(token), answer_tokens.count(token)) for token in common) / len(answer_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def answer_score(text: str, answers: list[str]) -> float:
    normalized = normalize_text(text)
    best = 0.0
    for answer in answers:
        answer_norm = normalize_text(answer)
        if answer_norm and answer_norm in normalized:
            return 1.0
        best = max(best, token_f1(text, answer))
    return best


def is_answer_match(text: str, answers: list[str], *, threshold: float = 0.35) -> bool:
    return answer_score(text, answers) >= threshold


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def build_raw_store(example: BenchmarkExample) -> tuple[LocalVectorStore, list[str]]:
    chunks = split_text(
        example.context,
        chunk_size=1200,
        chunk_overlap=200,
        source_path=f"longbench/{example.dataset}/{example.example_id}",
    )
    texts = [chunk.page_content for chunk in chunks]
    store = LocalVectorStore.build(
        texts=texts,
        metadatas=[chunk.metadata for chunk in chunks],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    return store, texts


def build_extractive_memory_store(example: BenchmarkExample) -> tuple[LocalVectorStore, list[dict[str, Any]]]:
    sections = split_text(
        example.context,
        chunk_size=3600,
        chunk_overlap=400,
        source_path=f"longbench/{example.dataset}/{example.example_id}",
    )
    records: list[dict[str, Any]] = []
    for section in sections:
        text = section.page_content
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        lead = " ".join(sentences[:3])[:850]
        facts = [
            sentence
            for sentence in sentences
            if re.search(r"\b[A-Z][a-z]+|\d", sentence)
        ][:8]
        formatted = (
            f"Section summary: {lead}\n"
            f"key_entities_and_facts: {'; '.join(facts)[:1200]}"
        )
        records.append(
            {
                "formatted_text": formatted,
                "raw_text": text,
                "metadata": section.metadata,
            }
        )
    store = LocalVectorStore.build(
        texts=[record["formatted_text"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    return store, records


def retrieve_raw(example: BenchmarkExample, *, raw_k: int) -> tuple[list[str], float]:
    start = time.time()
    store, _ = build_raw_store(example)
    docs = store.similarity_search(example.question, k=raw_k, embedding_model=DEFAULT_EMBEDDING_MODEL)
    return [doc.page_content for doc in docs], time.time() - start


def retrieve_extractive_memory(
    example: BenchmarkExample,
    *,
    summary_k: int,
    raw_k: int,
) -> tuple[list[str], float]:
    start = time.time()
    raw_store, _ = build_raw_store(example)
    memory_store, records = build_extractive_memory_store(example)
    memory_docs = memory_store.similarity_search(example.question, k=summary_k, embedding_model=DEFAULT_EMBEDDING_MODEL)
    raw_docs = raw_store.similarity_search(example.question, k=raw_k, embedding_model=DEFAULT_EMBEDDING_MODEL)

    memory_texts = [doc.page_content for doc in memory_docs]
    candidate_raw = [doc.page_content for doc in raw_docs]
    for doc in memory_docs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        for record in records:
            if str(record["metadata"].get("chunk_id", "")) == chunk_id:
                candidate_raw.append(str(record["raw_text"]))
                break

    deduped: list[str] = []
    seen: set[str] = set()
    for text in candidate_raw:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    support = sorted(deduped, key=lambda text: keyword_overlap_score(example.question, text), reverse=True)[:raw_k]
    return memory_texts + support, time.time() - start


def pack_context(items: list[str], max_context_chars: int) -> tuple[list[str], bool]:
    packed: list[str] = []
    used = 0
    truncated = False
    for item in items:
        text = item.strip()
        if not text:
            continue
        remaining = max_context_chars - used - (2 if packed else 0)
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            packed.append(text[:remaining].rstrip())
            truncated = True
            break
        packed.append(text)
        used += len(text) + (2 if packed else 0)
    return packed, truncated


def build_generation_messages(example: BenchmarkExample, context_items: list[str]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[Context {index}]\n{text}" for index, text in enumerate(context_items, 1))
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using only the provided context. "
                "Give a concise answer. If the answer is not supported, say I don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {example.question}\n\nAnswer:",
        },
    ]


def retrieve_memory(
    example: BenchmarkExample,
    *,
    memory_mode: str,
    summary_k: int,
    raw_k: int,
    summary_workers: int,
) -> tuple[str, list[str], float]:
    if memory_mode == "extractive":
        retrieved, latency = retrieve_extractive_memory(example, summary_k=summary_k, raw_k=raw_k)
        return "Summary Memory (extractive)", retrieved, latency
    if memory_mode == "structured":
        retrieved, latency = retrieve_structured_memory(
            example,
            summary_k=summary_k,
            raw_k=raw_k,
            max_workers=summary_workers,
        )
        return "Summary Memory (structured)", retrieved, latency
    raise ValueError(f"Unsupported memory mode: {memory_mode}")


def evaluate_retrieval(
    datasets: list[str],
    *,
    limit: int,
    raw_k: int,
    summary_k: int,
    memory_mode: str,
    summary_workers: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        examples = load_longbench_sample(dataset, limit=limit)
        for index, example in enumerate(examples, 1):
            print(f"[retrieval] {dataset} {index}/{len(examples)}")
            memory_method, memory_retrieved, memory_latency = retrieve_memory(
                example,
                memory_mode=memory_mode,
                summary_k=summary_k,
                raw_k=raw_k,
                summary_workers=summary_workers,
            )
            method_outputs = [
                ("Raw RAG", *retrieve_raw(example, raw_k=raw_k)),
                (memory_method, memory_retrieved, memory_latency),
            ]
            for method, retrieved, latency in method_outputs:
                context = "\n\n".join(retrieved)
                rows.append(
                    {
                        "dataset": dataset,
                        "example_id": example.example_id,
                        "method": method,
                        "question": example.question,
                        "answers": example.answers,
                        "hit": is_answer_match(context, example.answers),
                        "answer_score": answer_score(context, example.answers),
                        "context_chars": len(context),
                        "context_tokens_approx": approx_tokens(context),
                        "retrieved_records": len(retrieved),
                        "retrieval_latency": latency,
                    }
                )
    payload = {
        "datasets": datasets,
        "limit_per_dataset": limit,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "raw_k": raw_k,
        "summary_k": summary_k,
        "memory_mode": memory_mode,
        "summary": summarize_rows(rows, metric_key="hit"),
        "rows": rows,
    }
    return payload


def evaluate_generation(
    datasets: list[str],
    *,
    limit: int,
    raw_k: int,
    summary_k: int,
    memory_mode: str,
    summary_workers: int,
    max_context_chars: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        examples = load_longbench_sample(dataset, limit=limit)
        for index, example in enumerate(examples, 1):
            print(f"[generation] {dataset} {index}/{len(examples)}")
            memory_method, memory_retrieved, memory_latency = retrieve_memory(
                example,
                memory_mode=memory_mode,
                summary_k=summary_k,
                raw_k=raw_k,
                summary_workers=summary_workers,
            )
            method_outputs = [
                ("Raw RAG", *retrieve_raw(example, raw_k=raw_k)),
                (memory_method, memory_retrieved, memory_latency),
            ]
            for method, retrieved, retrieval_latency in method_outputs:
                packed, truncated = pack_context(retrieved, max_context_chars)
                try:
                    response = local_chat_completion(
                        build_generation_messages(example, packed),
                        max_tokens=max_tokens,
                        temperature=0.0,
                        timeout=timeout,
                        disable_thinking=True,
                    )
                    answer = response.text.strip()
                    error = ""
                except Exception as exc:
                    response = None
                    answer = f"[ERROR] {exc}"
                    error = str(exc)
                context = "\n\n".join(packed)
                rows.append(
                    {
                        "dataset": dataset,
                        "example_id": example.example_id,
                        "method": method,
                        "question": example.question,
                        "answers": example.answers,
                        "answer": answer,
                        "accuracy": is_answer_match(answer, example.answers),
                        "answer_score": answer_score(answer, example.answers),
                        "retrieval_hit": is_answer_match(context, example.answers),
                        "context_chars": len(context),
                        "context_tokens_approx": approx_tokens(context),
                        "context_truncated": truncated,
                        "retrieved_records": len(packed),
                        "retrieval_latency": retrieval_latency,
                        "generation_latency": response.latency if response else 0.0,
                        "prompt_tokens": response.prompt_tokens if response else 0,
                        "completion_tokens": response.completion_tokens if response else 0,
                        "total_tokens": response.total_tokens if response else 0,
                        "finish_reason": response.finish_reason if response else "error",
                        "error": error,
                    }
                )
                print(f"  {method}: {answer[:90]}")
    payload = {
        "datasets": datasets,
        "limit_per_dataset": limit,
        "local_model": local_model_name(),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "raw_k": raw_k,
        "summary_k": summary_k,
        "memory_mode": memory_mode,
        "max_context_chars": max_context_chars,
        "max_tokens": max_tokens,
        "summary": summarize_rows(rows, metric_key="accuracy"),
        "rows": rows,
    }
    return payload


def summarize_rows(rows: list[dict[str, Any]], *, metric_key: str) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    groups = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset, method in groups:
        subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
        total = len(subset)
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "score_name": metric_key,
                "score": sum(1 for row in subset if row.get(metric_key)) / total * 100 if total else 0.0,
                "correct_or_hits": sum(1 for row in subset if row.get(metric_key)),
                "total": total,
                "avg_answer_score": sum(float(row["answer_score"]) for row in subset) / total if total else 0.0,
                "avg_context_chars": sum(int(row["context_chars"]) for row in subset) / total if total else 0.0,
                "avg_context_tokens_approx": sum(int(row["context_tokens_approx"]) for row in subset) / total if total else 0.0,
                "avg_retrieval_latency": sum(float(row["retrieval_latency"]) for row in subset) / total if total else 0.0,
                "avg_generation_latency": (
                    sum(float(row.get("generation_latency", 0.0)) for row in subset) / total if total else 0.0
                ),
            }
        )
    return summary


def write_payload(payload: dict[str, Any], path: Path, csv_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "method",
            "score_name",
            "score",
            "correct_or_hits",
            "total",
            "avg_answer_score",
            "avg_context_chars",
            "avg_context_tokens_approx",
            "avg_retrieval_latency",
            "avg_generation_latency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["summary"]:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sampled LongBench datasets locally.")
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_LONGBENCH_DATASETS))
    parser.add_argument("--retrieval-limit", type=int, default=50)
    parser.add_argument("--generation-limit", type=int, default=10)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--raw-k", type=int, default=4)
    parser.add_argument("--summary-k", type=int, default=3)
    parser.add_argument("--memory-mode", choices=["extractive", "structured"], default="extractive")
    parser.add_argument("--summary-workers", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    if not args.skip_retrieval:
        retrieval_payload = evaluate_retrieval(
            args.datasets,
            limit=args.retrieval_limit,
            raw_k=args.raw_k,
            summary_k=args.summary_k,
            memory_mode=args.memory_mode,
            summary_workers=args.summary_workers,
        )
        write_payload(retrieval_payload, RETRIEVAL_JSON_PATH, RETRIEVAL_CSV_PATH)
        print(f"Saved {RETRIEVAL_JSON_PATH}")
        print(f"Saved {RETRIEVAL_CSV_PATH}")

    if not args.skip_generation:
        generation_payload = evaluate_generation(
            args.datasets,
            limit=args.generation_limit,
            raw_k=args.raw_k,
            summary_k=args.summary_k,
            memory_mode=args.memory_mode,
            summary_workers=args.summary_workers,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        write_payload(generation_payload, GENERATION_JSON_PATH, GENERATION_CSV_PATH)
        print(f"Saved {GENERATION_JSON_PATH}")
        print(f"Saved {GENERATION_CSV_PATH}")


if __name__ == "__main__":
    main()
