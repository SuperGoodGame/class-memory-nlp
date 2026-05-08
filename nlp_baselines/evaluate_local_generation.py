from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from .embeddings_utils import DEFAULT_EMBEDDING_MODEL
from .eval_utils import check_accuracy, check_hit
from .local_generation import classify_nli, local_chat_completion, local_model_name
from .query_data import CHROMA_PATH as RAW_CHROMA_PATH
from .query_data import build_db as build_raw_db
from .query_data import retrieve as retrieve_raw
from .query_no_memory import EVAL_DATASET
from .query_summary_memory import (
    SUMMARY_CACHE_PATH,
    SUMMARY_CHROMA_PATH,
    collect_support_chunks,
    ensure_summary_memory,
    retrieve_summaries,
)


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
DEFAULT_JSON_PATH = RESULTS_DIR / "local_generation_gemma4_aliceqa.json"
DEFAULT_CSV_PATH = TABLES_DIR / "generation_summary.csv"


def ensure_inputs() -> dict[str, str]:
    if not os.path.exists(RAW_CHROMA_PATH):
        print("  Raw vector index missing. Building it locally ...")
        build_raw_db()
    if not os.path.exists(SUMMARY_CACHE_PATH):
        raise RuntimeError(
            f"Missing {SUMMARY_CACHE_PATH}. Build summaries first; local generation does not "
            "silently call a remote summary API."
        )
    _, chunk_lookup = ensure_summary_memory(
        refresh_summaries=False,
        rebuild_summary_db=not os.path.exists(SUMMARY_CHROMA_PATH),
    )
    return chunk_lookup


def build_raw_messages(query: str, retrieved: list[str]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[Context {index}]\n{text}" for index, text in enumerate(retrieved, 1))
    return [
        {
            "role": "system",
            "content": (
                "You answer reading-comprehension questions. Use only the provided context. "
                "Answer briefly and directly. If the answer is unsupported, say I don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        },
    ]


def build_summary_messages(
    query: str,
    summary_docs: list[Any],
    support_chunks: list[str],
) -> list[dict[str, str]]:
    summary_context = "\n\n".join(
        f"[Memory {index}]\n{doc.page_content}" for index, doc in enumerate(summary_docs, 1)
    )
    support_context = "\n\n".join(
        f"[Support {index}]\n{text}" for index, text in enumerate(support_chunks, 1)
    )
    return [
        {
            "role": "system",
            "content": (
                "You answer reading-comprehension questions with structured memory and supporting excerpts. "
                "Use supporting excerpts for exact details. Answer briefly and directly. "
                "If the answer is unsupported, say I don't know."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Structured memory:\n{summary_context}\n\n"
                f"Supporting excerpts:\n{support_context}\n\n"
                "Answer:"
            ),
        },
    ]


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def pack_context_items(items: list[str], *, max_context_chars: int) -> tuple[list[str], bool]:
    if max_context_chars <= 0:
        return items, False

    packed: list[str] = []
    used = 0
    truncated = False
    separator_chars = 2
    for item in items:
        text = item.strip()
        if not text:
            continue
        remaining = max_context_chars - used - (separator_chars if packed else 0)
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            packed.append(text[:remaining].rstrip())
            truncated = True
            break
        packed.append(text)
        used += len(text) + (separator_chars if packed else 0)
    return packed, truncated


def run_method(
    *,
    method: str,
    query: str,
    expected_keywords: list[str],
    expected_in_docs: list[str],
    context_items: list[str],
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    nli_groundedness: bool,
    context_truncated: bool,
) -> dict[str, Any]:
    context = "\n\n".join(context_items)
    started = time.time()
    response = local_chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=timeout,
        disable_thinking=True,
    )
    total_latency = time.time() - started
    answer = response.text.strip()
    accuracy = check_accuracy(answer, expected_keywords)
    retrieval_hit = check_hit(context_items, expected_in_docs)
    row: dict[str, Any] = {
        "method": method,
        "answer": answer,
        "accuracy": accuracy,
        "retrieval_hit": retrieval_hit,
        "context_chars": len(context),
        "context_tokens_approx": approx_tokens(context),
        "retrieved_records": len(context_items),
        "context_truncated": context_truncated,
        "generation_latency": response.latency,
        "total_latency": total_latency,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
        "finish_reason": response.finish_reason,
        "empty_content_fallback": response.empty_content_fallback,
    }
    if nli_groundedness and answer:
        nli = classify_nli(
            premise=context[:12000],
            hypothesis=f"The answer to the question is: {answer}",
            timeout=timeout,
        )
        row["nli_groundedness"] = nli.label
        row["nli_source"] = nli.source
        row["nli_latency"] = nli.response.latency
    return row


def evaluate(
    *,
    limit: int | None,
    start_id: int | None,
    end_id: int | None,
    raw_k: int,
    summary_k: int,
    max_support_chunks: int,
    max_tokens: int,
    max_context_chars: int,
    timeout: int,
    nli_groundedness: bool,
) -> dict[str, Any]:
    chunk_lookup = ensure_inputs()
    indexed_cases = list(enumerate(EVAL_DATASET, 1))
    if start_id is not None:
        indexed_cases = [(idx, case) for idx, case in indexed_cases if idx >= start_id]
    if end_id is not None:
        indexed_cases = [(idx, case) for idx, case in indexed_cases if idx <= end_id]
    if limit:
        indexed_cases = indexed_cases[:limit]
    rows: list[dict[str, Any]] = []
    methods = ["Raw RAG", "Summary Memory"]
    aggregates: dict[str, dict[str, Any]] = {
        method: {
            "correct": 0,
            "retrieval_hits": 0,
            "empty_answers": 0,
            "total_context_chars": 0,
            "total_context_tokens_approx": 0,
            "total_generation_latency": 0.0,
            "total_latency": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "token_rows": 0,
            "errors": [],
        }
        for method in methods
    }

    for position, (index, testcase) in enumerate(indexed_cases, 1):
        expected_in_docs = getattr(testcase, "expected_in_docs", testcase.expected_in_answer)
        print(f"[{position}/{len(indexed_cases)}] #{index} {testcase.description}")

        raw_retrieved, _ = retrieve_raw(testcase.query, k=raw_k)
        summary_docs = retrieve_summaries(testcase.query, k=summary_k)
        support_chunks = collect_support_chunks(
            testcase.query,
            summary_docs,
            chunk_lookup,
            raw_k=raw_k,
            max_support_chunks=max_support_chunks,
        )
        method_inputs = [
            (
                "Raw RAG",
                raw_retrieved,
            ),
            (
                "Summary Memory",
                [doc.page_content for doc in summary_docs] + support_chunks,
            ),
        ]

        for method, context_items in method_inputs:
            packed_items, context_truncated = pack_context_items(
                context_items,
                max_context_chars=max_context_chars,
            )
            if method == "Raw RAG":
                messages = build_raw_messages(testcase.query, packed_items)
            else:
                packed_summary_docs = []
                packed_support_chunks: list[str] = []
                summary_texts = {doc.page_content for doc in summary_docs}
                for item in packed_items:
                    if item in summary_texts:
                        matched = next(doc for doc in summary_docs if doc.page_content == item)
                        packed_summary_docs.append(matched)
                    else:
                        packed_support_chunks.append(item)
                messages = build_summary_messages(
                    testcase.query,
                    packed_summary_docs,
                    packed_support_chunks,
                )
            try:
                row = run_method(
                    method=method,
                    query=testcase.query,
                    expected_keywords=testcase.expected_in_answer,
                    expected_in_docs=expected_in_docs,
                    context_items=packed_items,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    nli_groundedness=nli_groundedness,
                    context_truncated=context_truncated,
                )
            except Exception as exc:
                row = {
                    "method": method,
                    "answer": f"[ERROR] {exc}",
                    "accuracy": False,
                    "retrieval_hit": False,
                    "context_chars": len("\n\n".join(packed_items)),
                    "context_tokens_approx": approx_tokens("\n\n".join(packed_items)),
                    "retrieved_records": len(packed_items),
                    "context_truncated": context_truncated,
                    "generation_latency": 0.0,
                    "total_latency": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "finish_reason": "error",
                    "empty_content_fallback": None,
                }
                aggregates[method]["errors"].append(str(exc))

            row.update(
                {
                    "id": index,
                    "query": testcase.query,
                    "description": testcase.description,
                    "expected_keywords": testcase.expected_in_answer,
                    "expected_in_docs": expected_in_docs,
                }
            )
            rows.append(row)
            aggregate = aggregates[method]
            aggregate["correct"] += int(bool(row["accuracy"]))
            aggregate["retrieval_hits"] += int(bool(row["retrieval_hit"]))
            aggregate["empty_answers"] += int(not str(row["answer"]).strip())
            aggregate["total_context_chars"] += int(row["context_chars"])
            aggregate["total_context_tokens_approx"] += int(row["context_tokens_approx"])
            aggregate["total_generation_latency"] += float(row["generation_latency"])
            aggregate["total_latency"] += float(row["total_latency"])
            prompt_tokens = row.get("prompt_tokens")
            if prompt_tokens:
                aggregate["total_prompt_tokens"] += int(prompt_tokens)
                aggregate["total_completion_tokens"] += int(row.get("completion_tokens") or 0)
                aggregate["total_tokens"] += int(row.get("total_tokens") or 0)
                aggregate["token_rows"] += 1

            status = "OK" if row["accuracy"] else "MISS"
            print(f"  {method:<15} {status} {row['generation_latency']:.2f}s {row['answer'][:90]}")

    summary = []
    total = len(indexed_cases)
    for method in methods:
        aggregate = aggregates[method]
        token_rows = aggregate["token_rows"] or 1
        summary.append(
            {
                "dataset": "AliceQA-61" if limit is None else f"AliceQA-{total}",
                "local_model": local_model_name(),
                "method": method,
                "accuracy": aggregate["correct"] / total * 100 if total else 0.0,
                "correct": aggregate["correct"],
                "total": total,
                "retrieval_hit_rate": aggregate["retrieval_hits"] / total * 100 if total else 0.0,
                "retrieval_hits": aggregate["retrieval_hits"],
                "avg_context_chars": aggregate["total_context_chars"] / total if total else 0.0,
                "avg_context_tokens_approx": aggregate["total_context_tokens_approx"] / total if total else 0.0,
                "avg_generation_latency": aggregate["total_generation_latency"] / total if total else 0.0,
                "avg_total_latency": aggregate["total_latency"] / total if total else 0.0,
                "avg_prompt_tokens": aggregate["total_prompt_tokens"] / token_rows,
                "avg_completion_tokens": aggregate["total_completion_tokens"] / token_rows,
                "avg_total_tokens": aggregate["total_tokens"] / token_rows,
                "empty_answers": aggregate["empty_answers"],
                "errors": aggregate["errors"],
            }
        )

    return {
        "dataset": "AliceQA-61" if total == len(EVAL_DATASET) else f"AliceQA-subset-{total}",
        "total_questions": total,
        "local_model": local_model_name(),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "raw_k": raw_k,
        "summary_k": summary_k,
        "max_support_chunks": max_support_chunks,
        "max_tokens": max_tokens,
        "max_context_chars": max_context_chars,
        "disable_thinking": True,
        "nli_groundedness": nli_groundedness,
        "summary": summary,
        "rows": rows,
    }


def recompute_summary(payload: dict[str, Any]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in payload["rows"]})
    total = len({int(row["id"]) for row in payload["rows"]})
    summary = []
    for method in methods:
        rows = [row for row in payload["rows"] if row["method"] == method]
        token_rows = [row for row in rows if row.get("prompt_tokens")]
        errors = [
            str(row["answer"])[len("[ERROR] ") :]
            for row in rows
            if str(row.get("answer", "")).startswith("[ERROR]")
        ]
        denom_tokens = len(token_rows) or 1
        summary.append(
            {
                "dataset": "AliceQA-61" if total == len(EVAL_DATASET) else f"AliceQA-subset-{total}",
                "local_model": payload["local_model"],
                "method": method,
                "accuracy": sum(1 for row in rows if row["accuracy"]) / total * 100 if total else 0.0,
                "correct": sum(1 for row in rows if row["accuracy"]),
                "total": total,
                "retrieval_hit_rate": sum(1 for row in rows if row["retrieval_hit"]) / total * 100 if total else 0.0,
                "retrieval_hits": sum(1 for row in rows if row["retrieval_hit"]),
                "avg_context_chars": sum(int(row["context_chars"]) for row in rows) / total if total else 0.0,
                "avg_context_tokens_approx": sum(int(row["context_tokens_approx"]) for row in rows) / total if total else 0.0,
                "avg_generation_latency": sum(float(row["generation_latency"]) for row in rows) / total if total else 0.0,
                "avg_total_latency": sum(float(row["total_latency"]) for row in rows) / total if total else 0.0,
                "avg_prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in token_rows) / denom_tokens,
                "avg_completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in token_rows) / denom_tokens,
                "avg_total_tokens": sum(int(row.get("total_tokens") or 0) for row in token_rows) / denom_tokens,
                "empty_answers": sum(1 for row in rows if not str(row.get("answer", "")).strip()),
                "errors": errors,
            }
        )
    method_order = {"Raw RAG": 0, "Summary Memory": 1}
    payload["summary"] = sorted(summary, key=lambda row: method_order.get(row["method"], 99))
    payload["total_questions"] = total
    payload["dataset"] = "AliceQA-61" if total == len(EVAL_DATASET) else f"AliceQA-subset-{total}"
    payload["rows"] = sorted(payload["rows"], key=lambda row: (int(row["id"]), str(row["method"])))
    return payload


def merge_payload(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    replacement_keys = {(int(row["id"]), str(row["method"])) for row in update["rows"]}
    kept_rows = [
        row
        for row in existing.get("rows", [])
        if (int(row["id"]), str(row["method"])) not in replacement_keys
    ]
    merged = dict(existing)
    for key in [
        "local_model",
        "embedding_model",
        "raw_k",
        "summary_k",
        "max_support_chunks",
        "max_tokens",
        "max_context_chars",
        "disable_thinking",
        "nli_groundedness",
    ]:
        merged[key] = update.get(key, existing.get(key))
    merged["rows"] = kept_rows + update["rows"]
    return recompute_summary(merged)


def write_outputs(payload: dict[str, Any], *, merge_existing: bool = False) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if merge_existing and DEFAULT_JSON_PATH.exists():
        existing = json.loads(DEFAULT_JSON_PATH.read_text(encoding="utf-8"))
        payload = merge_payload(existing, payload)
    else:
        payload = recompute_summary(payload)
    DEFAULT_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with DEFAULT_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "local_model",
            "method",
            "accuracy",
            "correct",
            "total",
            "retrieval_hit_rate",
            "retrieval_hits",
            "avg_context_chars",
            "avg_context_tokens_approx",
            "avg_generation_latency",
            "avg_total_latency",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "empty_answers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["summary"]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return payload


def print_summary(payload: dict[str, Any]) -> None:
    print("\nLocal Gemma4 generation evaluation")
    print(f"Dataset: {payload['dataset']}")
    print(f"Model: {payload['local_model']}")
    print(f"Thinking disabled: {payload['disable_thinking']}")
    print()
    print(f"{'Method':<18} {'Accuracy':>10} {'Hit Rate':>10} {'Avg Tokens':>11} {'Latency':>10}")
    print("-" * 67)
    for row in payload["summary"]:
        accuracy = f"{row['accuracy']:.1f}%"
        hit_rate = f"{row['retrieval_hit_rate']:.1f}%"
        tokens = f"{row['avg_total_tokens']:.1f}"
        latency = f"{row['avg_generation_latency']:.2f}s"
        print(f"{row['method']:<18} {accuracy:>10} {hit_rate:>10} {tokens:>11} {latency:>10}")
    print(f"\nSaved {DEFAULT_JSON_PATH}")
    print(f"Appended {DEFAULT_CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Gemma4 generation evaluation.")
    parser.add_argument("--dataset", default="aliceqa", choices=["aliceqa"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--raw-k", type=int, default=4)
    parser.add_argument("--summary-k", type=int, default=3)
    parser.add_argument("--max-support-chunks", type=int, default=6)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--nli-groundedness", action="store_true")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    payload = evaluate(
        limit=args.limit,
        start_id=args.start_id,
        end_id=args.end_id,
        raw_k=args.raw_k,
        summary_k=args.summary_k,
        max_support_chunks=args.max_support_chunks,
        max_tokens=args.max_tokens,
        max_context_chars=args.max_context_chars,
        timeout=args.timeout,
        nli_groundedness=args.nli_groundedness,
    )
    payload = write_outputs(payload, merge_existing=args.merge_existing)
    print_summary(payload)


if __name__ == "__main__":
    main()
