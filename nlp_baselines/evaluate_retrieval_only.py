from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_utils import load_text
from .embeddings_utils import DEFAULT_EMBEDDING_MODEL
from .eval_utils import check_hit
from .query_data import CHROMA_PATH as RAW_CHROMA_PATH
from .query_data import build_db as build_raw_db
from .query_data import retrieve as retrieve_raw
from .query_no_memory import DATA_PATH as ALICE_DATA_PATH
from .query_no_memory import EVAL_DATASET
from .query_summary_memory import (
    RAW_CHROMA_PATH as SUMMARY_RAW_CHROMA_PATH,
    SUMMARY_CACHE_PATH,
    SUMMARY_CHROMA_PATH,
    collect_support_chunks,
    compute_compression_ratio,
    ensure_summary_memory,
    retrieve_summaries,
)


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
DEFAULT_JSON_PATH = RESULTS_DIR / "aliceqa_retrieval_only.json"
DEFAULT_CSV_PATH = TABLES_DIR / "retrieval_summary.csv"


@dataclass
class RetrievalStats:
    method: str
    hits: int
    total: int
    avg_context_chars: float
    avg_context_tokens_approx: float
    avg_retrieved_records: float
    avg_retrieval_latency: float
    rows: list[dict[str, Any]]

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total * 100


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def ensure_inputs(*, allow_summary_build: bool = False) -> tuple[list[dict[str, object]], dict[str, str]]:
    if not os.path.exists(RAW_CHROMA_PATH):
        print("  Raw vector index missing. Building it locally ...")
        build_raw_db()

    if not os.path.exists(SUMMARY_CACHE_PATH):
        if not allow_summary_build:
            raise RuntimeError(
                f"Missing {SUMMARY_CACHE_PATH}. Re-run with --allow-summary-build to generate "
                "summaries through the configured chat API."
            )
    if not os.path.exists(SUMMARY_CHROMA_PATH):
        if not os.path.exists(SUMMARY_CACHE_PATH):
            raise RuntimeError(f"Missing {SUMMARY_CACHE_PATH}; cannot build summary index.")
        print("  Summary vector index missing. Rebuilding from cached summaries ...")

    return ensure_summary_memory(
        refresh_summaries=False,
        rebuild_summary_db=not os.path.exists(SUMMARY_CHROMA_PATH),
    )


def evaluate_raw_rag(*, k_values: list[int]) -> RetrievalStats:
    rows: list[dict[str, Any]] = []
    hits = 0
    total_chars = 0
    total_records = 0
    total_latency = 0.0
    main_k = max(k_values)

    for index, testcase in enumerate(EVAL_DATASET, 1):
        expected_in_docs = getattr(testcase, "expected_in_docs", testcase.expected_in_answer)
        start = time.time()
        retrieved, _ = retrieve_raw(testcase.query, k=main_k)
        latency = time.time() - start
        context = "\n\n".join(retrieved)
        is_hit = check_hit(retrieved, expected_in_docs)
        if is_hit:
            hits += 1
        total_chars += len(context)
        total_records += len(retrieved)
        total_latency += latency
        recall = {
            f"recall_at_{k}": check_hit(retrieved[:k], expected_in_docs)
            for k in k_values
        }
        rows.append(
            {
                "id": index,
                "query": testcase.query,
                "description": testcase.description,
                "expected": expected_in_docs,
                "hit": is_hit,
                "context_chars": len(context),
                "context_tokens_approx": approx_tokens(context),
                "retrieved_records": len(retrieved),
                "retrieval_latency": latency,
                **recall,
            }
        )

    total = len(EVAL_DATASET)
    return RetrievalStats(
        method="Raw RAG",
        hits=hits,
        total=total,
        avg_context_chars=total_chars / total if total else 0,
        avg_context_tokens_approx=(total_chars / 4) / total if total else 0,
        avg_retrieved_records=total_records / total if total else 0,
        avg_retrieval_latency=total_latency / total if total else 0,
        rows=rows,
    )


def evaluate_summary_memory(
    *,
    chunk_lookup: dict[str, str],
    summary_k: int,
    raw_k: int,
    max_support_chunks: int,
    k_values: list[int],
) -> RetrievalStats:
    rows: list[dict[str, Any]] = []
    hits = 0
    total_chars = 0
    total_records = 0
    total_latency = 0.0

    for index, testcase in enumerate(EVAL_DATASET, 1):
        expected_in_docs = getattr(testcase, "expected_in_docs", testcase.expected_in_answer)
        start = time.time()
        summary_docs = retrieve_summaries(testcase.query, k=summary_k)
        support_chunks = collect_support_chunks(
            testcase.query,
            summary_docs,
            chunk_lookup,
            raw_k=raw_k,
            max_support_chunks=max_support_chunks,
        )
        latency = time.time() - start
        retrieved = [doc.page_content for doc in summary_docs] + support_chunks
        context = "\n\n".join(retrieved)
        is_hit = check_hit(retrieved, expected_in_docs)
        if is_hit:
            hits += 1
        total_chars += len(context)
        total_records += len(retrieved)
        total_latency += latency
        recall = {
            f"recall_at_{k}": check_hit(retrieved[:k], expected_in_docs)
            for k in k_values
        }
        rows.append(
            {
                "id": index,
                "query": testcase.query,
                "description": testcase.description,
                "expected": expected_in_docs,
                "hit": is_hit,
                "context_chars": len(context),
                "context_tokens_approx": approx_tokens(context),
                "retrieved_records": len(retrieved),
                "summary_records": len(summary_docs),
                "support_chunks": len(support_chunks),
                "retrieval_latency": latency,
                **recall,
            }
        )

    total = len(EVAL_DATASET)
    return RetrievalStats(
        method="Summary Memory",
        hits=hits,
        total=total,
        avg_context_chars=total_chars / total if total else 0,
        avg_context_tokens_approx=(total_chars / 4) / total if total else 0,
        avg_retrieved_records=total_records / total if total else 0,
        avg_retrieval_latency=total_latency / total if total else 0,
        rows=rows,
    )


def stat_to_summary(stats: RetrievalStats) -> dict[str, Any]:
    return {
        "method": stats.method,
        "hit_rate": stats.hit_rate,
        "hits": stats.hits,
        "total": stats.total,
        "avg_context_chars": stats.avg_context_chars,
        "avg_context_tokens_approx": stats.avg_context_tokens_approx,
        "avg_retrieved_records": stats.avg_retrieved_records,
        "avg_retrieval_latency": stats.avg_retrieval_latency,
    }


def write_outputs(
    *,
    raw_stats: RetrievalStats,
    summary_stats: RetrievalStats,
    records: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    document_text = load_text(ALICE_DATA_PATH)
    full_context_chars = len(document_text)
    payload = {
        "dataset": "AliceQA-61",
        "total_questions": len(EVAL_DATASET),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "raw_k": args.raw_k,
        "summary_k": args.summary_k,
        "max_support_chunks": args.max_support_chunks,
        "approx_token_rule": "characters / 4",
        "summary_compression_ratio": compute_compression_ratio(records),
        "full_context": {
            "context_chars_per_query": full_context_chars,
            "context_tokens_approx_per_query": approx_tokens(document_text),
        },
        "summary": [stat_to_summary(raw_stats), stat_to_summary(summary_stats)],
        "rows": {
            "raw_rag": raw_stats.rows,
            "summary_memory": summary_stats.rows,
        },
    }
    DEFAULT_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with DEFAULT_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "hit_rate",
                "hits",
                "total",
                "avg_context_chars",
                "avg_context_tokens_approx",
                "avg_retrieved_records",
                "avg_retrieval_latency",
            ],
        )
        writer.writeheader()
        for item in payload["summary"]:
            writer.writerow({"dataset": "AliceQA-61", **item})
        writer.writerow(
            {
                "dataset": "AliceQA-61",
                "method": "Full Context",
                "hit_rate": "",
                "hits": "",
                "total": len(EVAL_DATASET),
                "avg_context_chars": full_context_chars,
                "avg_context_tokens_approx": approx_tokens(document_text),
                "avg_retrieved_records": 1,
                "avg_retrieval_latency": "",
            }
        )
    return payload


def print_summary(payload: dict[str, Any]) -> None:
    print("\nRetrieval-only local evaluation")
    print("Dataset: AliceQA-61")
    print(f"Embedding: {payload['embedding_model']}")
    print(f"Summary compression ratio: {payload['summary_compression_ratio']:.3f}")
    print()
    print(f"{'Method':<18} {'Hit Rate':>10} {'Hits':>8} {'Avg Chars':>12} {'Avg Latency':>13}")
    print("-" * 68)
    for row in payload["summary"]:
        hits = f"{row['hits']}/{row['total']}"
        print(
            f"{row['method']:<18} "
            f"{row['hit_rate']:>9.1f}% "
            f"{hits:>8} "
            f"{row['avg_context_chars']:>12.1f} "
            f"{row['avg_retrieval_latency']:>12.3f}s"
        )
    full = payload["full_context"]
    print(
        f"{'Full Context':<18} "
        f"{'N/A':>10} "
        f"{'N/A':>8} "
        f"{full['context_chars_per_query']:>12.1f} "
        f"{'N/A':>13}"
    )
    print(f"\nSaved {DEFAULT_JSON_PATH}")
    print(f"Saved {DEFAULT_CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation without final LLM generation.")
    parser.add_argument("--summary-k", type=int, default=3)
    parser.add_argument("--raw-k", type=int, default=4)
    parser.add_argument("--max-support-chunks", type=int, default=6)
    parser.add_argument("--allow-summary-build", action="store_true")
    args = parser.parse_args()

    if SUMMARY_RAW_CHROMA_PATH != RAW_CHROMA_PATH:
        raise RuntimeError("Raw index path mismatch between raw and summary modules.")

    records, chunk_lookup = ensure_inputs(allow_summary_build=args.allow_summary_build)
    k_values = [1, 3, 5]
    raw_stats = evaluate_raw_rag(k_values=k_values)
    summary_stats = evaluate_summary_memory(
        chunk_lookup=chunk_lookup,
        summary_k=args.summary_k,
        raw_k=args.raw_k,
        max_support_chunks=args.max_support_chunks,
        k_values=k_values,
    )
    payload = write_outputs(raw_stats=raw_stats, summary_stats=summary_stats, records=records, args=args)
    print_summary(payload)


if __name__ == "__main__":
    main()
