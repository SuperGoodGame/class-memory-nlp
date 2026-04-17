from __future__ import annotations

import argparse
import json
import os
import shutil

from .api_utils import chat_completion, describe_chat_target
from .data_utils import TextChunk, load_text, split_text
from .embeddings_utils import DEFAULT_EMBEDDING_MODEL
from .eval_utils import check_accuracy, check_hit, pad
from .query_no_memory import EVAL_DATASET
from .query_data import CHROMA_PATH as RAW_CHROMA_PATH
from .query_data import build_db as build_raw_db
from .query_data import retrieve as retrieve_raw
from .summary_utils import (
    SUMMARY_SCHEMA,
    format_summary_record,
    keyword_overlap_score,
    parse_summary_output,
    summary_output_needs_retry,
)
from .vector_store import LocalVectorStore, RetrievedRecord


DATA_PATH = "data/books/alice_in_wonderland.md"
SUMMARY_CHROMA_PATH = "chroma_summary"
SUMMARY_CACHE_DIR = "summary_cache"
SUMMARY_CACHE_PATH = os.path.join(SUMMARY_CACHE_DIR, "alice_section_summaries.jsonl")
SUMMARY_CHUNK_SIZE = 4000
SUMMARY_CHUNK_OVERLAP = 400


def load_documents(
    *,
    chunk_size: int = SUMMARY_CHUNK_SIZE,
    chunk_overlap: int = SUMMARY_CHUNK_OVERLAP,
) -> list[TextChunk]:
    text = load_text(DATA_PATH)
    return split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source_path=DATA_PATH)


def build_chunk_lookup(chunks: list[TextChunk]) -> dict[str, str]:
    return {str(chunk.metadata["chunk_id"]): chunk.page_content for chunk in chunks}


def build_sections(
    chunks: list[TextChunk],
    *,
    section_size: int = 1,
    section_stride: int = 1,
) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    for start in range(0, len(chunks), section_stride):
        window = chunks[start : start + section_size]
        if not window:
            continue
        sections.append(
            {
                "section_id": f"section_{len(sections)}",
                "chunk_start": start,
                "chunk_end": start + len(window) - 1,
                "raw_chunk_ids": [chunk.metadata["chunk_id"] for chunk in window],
                "raw_text": "\n\n".join(chunk.page_content for chunk in window),
            }
        )
        if start + section_size >= len(chunks):
            break
    return sections


def build_summary_prompt(section_text: str) -> str:
    schema_text = json.dumps(SUMMARY_SCHEMA, ensure_ascii=True, indent=2)
    return (
        "Summarize the following narrative passage into structured long-term memory for QA.\n"
        "Requirements:\n"
        "1. Preserve exact details that matter for question answering: names, numbers, labels, locations, "
        "transformations, and short quoted phrases.\n"
        "2. Keep the summary concise but specific.\n"
        "3. Output valid JSON only using this exact schema:\n"
        f"{schema_text}\n\n"
        "Passage:\n"
        f"{section_text}"
    )


def summarize_section(section_text: str) -> dict[str, object]:
    candidate_segments = [
        section_text,
        section_text[:3000],
        section_text[:1800],
    ]
    last_error: Exception | None = None

    for index, candidate_text in enumerate(candidate_segments):
        if not candidate_text.strip():
            continue
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You build compact long-term memory for a reading comprehension system. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": build_summary_prompt(candidate_text)},
            ]
            response = chat_completion(messages, max_tokens=500, temperature=0.0, timeout=120)
            parsed = parse_summary_output(response.text)

            if summary_output_needs_retry(response.text, parsed):
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Your previous output was malformed or too vague. "
                            "Return valid JSON only and include exact facts."
                        ),
                    },
                    {"role": "user", "content": build_summary_prompt(candidate_text)},
                ]
                response = chat_completion(retry_messages, max_tokens=500, temperature=0.0, timeout=120)
                parsed = parse_summary_output(response.text)

            if index > 0 and not parsed.get("supporting_quotes"):
                parsed["supporting_quotes"] = [candidate_text[:160].strip()]
            return parsed
        except Exception as exc:
            last_error = exc

    fallback_text = section_text[:500].strip()
    fallback_summary = {
        "section_summary": fallback_text,
        "key_entities": [],
        "key_events": [],
        "exact_facts": [],
        "supporting_quotes": [fallback_text[:160]] if fallback_text else [],
    }
    if last_error is not None:
        fallback_summary["key_events"] = [f"summary_fallback_due_to_error: {last_error}"]
    return fallback_summary


def generate_summary_records(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, section in enumerate(sections, 1):
        print(f"  Summarizing section {index}/{len(sections)} ...")
        summary = summarize_section(str(section["raw_text"]))
        formatted = format_summary_record(summary)
        records.append(
            {
                "section_id": section["section_id"],
                "chunk_start": section["chunk_start"],
                "chunk_end": section["chunk_end"],
                "raw_chunk_ids": section["raw_chunk_ids"],
                "summary": summary,
                "formatted_text": formatted,
                "source_char_count": len(str(section["raw_text"])),
                "summary_char_count": len(formatted),
            }
        )
    return records


def save_summary_cache(records: list[dict[str, object]]) -> None:
    os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
    with open(SUMMARY_CACHE_PATH, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_summary_cache() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_summary_db(records: list[dict[str, object]]) -> None:
    if os.path.exists(SUMMARY_CHROMA_PATH):
        shutil.rmtree(SUMMARY_CHROMA_PATH)

    store = LocalVectorStore.build(
        texts=[str(record["formatted_text"]) for record in records],
        metadatas=[
            {
                "section_id": str(record["section_id"]),
                "chunk_start": int(record["chunk_start"]),
                "chunk_end": int(record["chunk_end"]),
                "raw_chunk_ids": list(record["raw_chunk_ids"]),
            }
            for record in records
        ],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    store.save(SUMMARY_CHROMA_PATH)
    print(f"  Saved {store.count()} summary sections to {SUMMARY_CHROMA_PATH}")


def ensure_summary_memory(
    *,
    refresh_summaries: bool = False,
    rebuild_summary_db: bool = False,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    chunks = load_documents()
    chunk_lookup = build_chunk_lookup(chunks)
    sections = build_sections(chunks)

    if refresh_summaries or not os.path.exists(SUMMARY_CACHE_PATH):
        print("  Generating section summaries through the remote API ...")
        records = generate_summary_records(sections)
        save_summary_cache(records)
    else:
        records = load_summary_cache()

    if rebuild_summary_db or not os.path.exists(SUMMARY_CHROMA_PATH):
        print("  Building summary vector index ...")
        build_summary_db(records)

    return records, chunk_lookup


def retrieve_summaries(query: str, *, k: int = 3) -> list[RetrievedRecord]:
    store = LocalVectorStore.load(SUMMARY_CHROMA_PATH)
    candidates = store.similarity_search(query, k=max(k * 3, k), embedding_model=DEFAULT_EMBEDDING_MODEL)
    rescored: list[tuple[float, RetrievedRecord]] = []
    for rank, doc in enumerate(candidates):
        score = (1.0 / (rank + 1)) + (0.5 * keyword_overlap_score(query, doc.page_content))
        rescored.append((score, doc))
    rescored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in rescored[:k]]


def collect_support_chunks(
    query: str,
    summary_docs: list[RetrievedRecord],
    chunk_lookup: dict[str, str],
    *,
    raw_k: int = 4,
    max_support_chunks: int = 6,
) -> list[str]:
    direct_hits, _ = retrieve_raw(query, k=raw_k)
    candidates: list[str] = list(direct_hits)

    for doc in summary_docs:
        raw_chunk_ids = doc.metadata.get("raw_chunk_ids", [])
        for chunk_id in raw_chunk_ids:
            chunk_text = chunk_lookup.get(str(chunk_id))
            if chunk_text:
                candidates.append(chunk_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    ranked = sorted(deduped, key=lambda text: keyword_overlap_score(query, text), reverse=True)
    return ranked[:max_support_chunks]


def build_answer_messages(
    query: str,
    summary_docs: list[RetrievedRecord],
    support_chunks: list[str],
) -> list[dict[str, str]]:
    summary_context = "\n\n".join(f"[Memory {index}] {doc.page_content}" for index, doc in enumerate(summary_docs, 1))
    support_context = "\n\n".join(f"[Support {index}] {text}" for index, text in enumerate(support_chunks, 1))
    return [
        {
            "role": "system",
            "content": (
                "You are a QA assistant with structured long-term memory. "
                "Use the summary memory for high-level recall and the supporting excerpts for exact details. "
                "Prefer the most specific grounded description available. "
                "If summary memory contains a precise named object, place, or label and a supporting excerpt only "
                "gives a more generic description of the same scene, keep the precise named description. "
                "Otherwise, use the supporting excerpts for exact details. "
                "If unsupported, say 'I don't know.' "
                "Answer briefly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Structured summary memory:\n{summary_context}\n\n"
                f"Supporting excerpts:\n{support_context}"
            ),
        },
    ]


def run_summary_memory(
    query: str,
    chunk_lookup: dict[str, str],
    *,
    summary_k: int = 3,
    raw_k: int = 4,
):
    summary_docs = retrieve_summaries(query, k=summary_k)
    support_chunks = collect_support_chunks(query, summary_docs, chunk_lookup, raw_k=raw_k)
    response = chat_completion(
        build_answer_messages(query, summary_docs, support_chunks),
        max_tokens=256,
        temperature=0.0,
        timeout=120,
    )
    retrieved_blobs = [doc.page_content for doc in summary_docs] + support_chunks
    return response, summary_docs, support_chunks, retrieved_blobs


def compute_compression_ratio(records: list[dict[str, object]]) -> float:
    source_chars = sum(int(record.get("source_char_count", 0)) for record in records)
    summary_chars = sum(int(record.get("summary_char_count", 0)) for record in records)
    if source_chars == 0:
        return 0.0
    return summary_chars / source_chars


def run_evaluation(
    *,
    summary_k: int = 3,
    raw_k: int = 4,
    refresh_summaries: bool = False,
    rebuild_summary_db: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    if not os.path.exists(RAW_CHROMA_PATH):
        print("  Raw baseline index missing. Building it first ...")
        build_raw_db()

    records, chunk_lookup = ensure_summary_memory(
        refresh_summaries=refresh_summaries,
        rebuild_summary_db=rebuild_summary_db,
    )
    compression_ratio = compute_compression_ratio(records)

    print("\n" + "=" * 92)
    print("  Summary-Memory Evaluation  —  alice_in_wonderland.md")
    print("=" * 92)
    print(f"  Chat API     : {describe_chat_target()}")
    print(f"  Embedding    : {DEFAULT_EMBEDDING_MODEL}")
    print(f"  Summary top-k: {summary_k}")
    print(f"  Raw top-k    : {raw_k}")
    print(f"  Compression  : {compression_ratio:.3f}")
    print("=" * 92 + "\n")

    total = len(EVAL_DATASET)
    hits = 0
    correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_all_tokens = 0
    total_latency = 0.0
    error_messages: list[str] = []
    rows = []

    for index, testcase in enumerate(EVAL_DATASET, 1):
        expected_in_docs = getattr(testcase, "expected_in_docs", testcase.expected_in_answer)
        try:
            response, summary_docs, support_chunks, retrieved_blobs = run_summary_memory(
                testcase.query,
                chunk_lookup,
                summary_k=summary_k,
                raw_k=raw_k,
            )
            answer = response.text
            prompt_tokens = response.prompt_tokens or 0
            completion_tokens = response.completion_tokens or 0
            total_tokens = response.total_tokens or 0
            latency = response.latency
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            summary_docs = []
            support_chunks = []
            retrieved_blobs = []
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            latency = 0.0
            error_messages.append(str(exc))

        is_hit = check_hit(retrieved_blobs, expected_in_docs)
        is_correct = check_accuracy(answer, testcase.expected_in_answer)

        if is_hit:
            hits += 1
        if is_correct:
            correct += 1

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_all_tokens += total_tokens
        total_latency += latency

        rows.append(
            {
                "id": index,
                "query": testcase.query,
                "hit": "✓" if is_hit else "✗",
                "acc": "✓" if is_correct else "✗",
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency": latency,
                "summary_docs": [doc.page_content for doc in summary_docs],
                "support_chunks": support_chunks,
            }
        )

        if verbose:
            print(f"[{index}/{total}] {testcase.query}")
            print(f"  Expected (docs) : {', '.join(expected_in_docs)}")
            print(f"  Expected (ans)  : {', '.join(testcase.expected_in_answer)}")
            for doc_index, doc in enumerate(summary_docs, 1):
                print(f"  Memory {doc_index}: {doc.page_content[:140]}...")
            for chunk_index, chunk in enumerate(support_chunks, 1):
                print(f"  Support {chunk_index}: {chunk[:140]}...")
            print(f"  Answer      : {answer}")
            print(f"  Hit         : {'✓' if is_hit else '✗'}")
            print(f"  Accuracy    : {'✓' if is_correct else '✗'}")
            print(f"  Prompt tok  : {prompt_tokens}")
            print(f"  Compl tok   : {completion_tokens}")
            print(f"  Total tok   : {total_tokens}")
            print(f"  Latency     : {latency:.2f}s")
            print()

    hit_rate = hits / total * 100
    accuracy = correct / total * 100
    avg_prompt_tokens = total_prompt_tokens / total if total else 0
    avg_completion_tokens = total_completion_tokens / total if total else 0
    avg_total_tokens = total_all_tokens / total if total else 0
    avg_latency = total_latency / total if total else 0

    query_width = 28
    table_width = query_width + 8 + 12 + 10 + 3

    print("┌" + "─" * table_width + "┐")
    header = f"│ {'#':^3}  {'Query':<{query_width}}  {'Hit':^4}  {'Acc':^4}  {'Tokens':^8}  {'Latency':^8} │"
    print(header)
    print("├" + "─" * table_width + "┤")
    for row in rows:
        latency_text = f"{row['latency']:.2f}s"
        line = (
            f"│ {pad(str(row['id']),3)}  "
            f"{pad(row['query'],query_width)}  "
            f"{pad(row['hit'],4)}  "
            f"{pad(row['acc'],4)}  "
            f"{pad(row['total_tokens'],8, '>')}  "
            f"{pad(latency_text,8, '>')} │"
        )
        print(line)
    print("├" + "─" * table_width + "┤")
    print(f"│  Hit Rate           :  {hit_rate:5.1f}%  ({hits}/{total})" + " " * (table_width - 35) + "│")
    print(f"│  Accuracy           :  {accuracy:5.1f}%  ({correct}/{total})" + " " * (table_width - 35) + "│")
    print(f"│  Avg Prompt Tokens  :  {avg_prompt_tokens:8.1f}" + " " * (table_width - 35) + "│")
    print(f"│  Avg Completion Tok :  {avg_completion_tokens:8.1f}" + " " * (table_width - 35) + "│")
    print(f"│  Avg Total Tokens   :  {avg_total_tokens:8.1f}" + " " * (table_width - 35) + "│")
    print(f"│  Avg Latency        :  {avg_latency:8.2f}s" + " " * (table_width - 35) + "│")
    print(f"│  Compression Ratio  :  {compression_ratio:8.3f}" + " " * (table_width - 35) + "│")
    print("└" + "─" * table_width + "┘")
    if error_messages:
        print(f"  Errors             : {len(error_messages)}")
        print(f"  First error        : {error_messages[0]}")

    return {
        "hit_rate": hit_rate,
        "accuracy": accuracy,
        "hits": hits,
        "correct": correct,
        "total": total,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_latency": avg_latency,
        "compression_ratio": compression_ratio,
        "errors": error_messages,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summary-memory baseline on alice_in_wonderland.md")
    parser.add_argument("query_text", type=str, nargs="?", help="Single query.")
    parser.add_argument("--eval", action="store_true", help="Run the summary evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details.")
    parser.add_argument("--summary-k", type=int, default=3, help="Number of summary memories to retrieve.")
    parser.add_argument("--raw-k", type=int, default=4, help="Number of raw support chunks to retrieve.")
    parser.add_argument("--rebuild-raw", action="store_true", help="Rebuild the raw vector index.")
    parser.add_argument("--rebuild-summary", action="store_true", help="Rebuild the summary vector index.")
    parser.add_argument("--refresh-summaries", action="store_true", help="Regenerate section summaries via the API.")
    args = parser.parse_args()

    if args.rebuild_raw and os.path.exists(RAW_CHROMA_PATH):
        shutil.rmtree(RAW_CHROMA_PATH)
        print(f"  Removed {RAW_CHROMA_PATH}")

    if args.rebuild_summary and os.path.exists(SUMMARY_CHROMA_PATH):
        shutil.rmtree(SUMMARY_CHROMA_PATH)
        print(f"  Removed {SUMMARY_CHROMA_PATH}")

    if not os.path.exists(RAW_CHROMA_PATH):
        print("  Building raw baseline vector index ...")
        build_raw_db()

    _, chunk_lookup = ensure_summary_memory(
        refresh_summaries=args.refresh_summaries,
        rebuild_summary_db=args.rebuild_summary,
    )

    if args.eval:
        run_evaluation(
            summary_k=args.summary_k,
            raw_k=args.raw_k,
            refresh_summaries=args.refresh_summaries,
            rebuild_summary_db=args.rebuild_summary,
            verbose=args.verbose,
        )
    elif args.query_text:
        response, summary_docs, support_chunks, _ = run_summary_memory(
            args.query_text,
            chunk_lookup,
            summary_k=args.summary_k,
            raw_k=args.raw_k,
        )
        print("\n=== Retrieved Summary Memories ===")
        for index, doc in enumerate(summary_docs, 1):
            print(f"\n[{index}] score={doc.score:.4f}\n{doc.page_content}\n")
        print("\n=== Supporting Raw Excerpts ===")
        for index, chunk in enumerate(support_chunks, 1):
            print(f"\n[{index}]\n{chunk}\n")
        print("\n=== Prompt Length Stats ===")
        print(f"Prompt Tokens     : {response.prompt_tokens}")
        print(f"Completion Tokens : {response.completion_tokens}")
        print(f"Total Tokens      : {response.total_tokens}")
        print(f"Latency           : {response.latency:.2f}s")
        print("\n=== Model Answer ===")
        print(response.text)
    else:
        print("Usage:")
        print('  python query_summary_memory.py "question"')
        print("  python query_summary_memory.py --eval")
        print("  python query_summary_memory.py --refresh-summaries --rebuild-summary")


if __name__ == "__main__":
    main()
