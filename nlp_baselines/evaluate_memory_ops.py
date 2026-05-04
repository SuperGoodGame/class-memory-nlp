from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .api_utils import chat_completion, describe_chat_target
from .data_utils import split_text
from .embeddings_utils import DEFAULT_EMBEDDING_MODEL
from .vector_store import LocalVectorStore


DATA_PATH = Path("data/memory_ops/memory_operations.json")


@dataclass
class MemoryOpsCase:
    id: str
    operation: str
    history: list[str]
    question: str
    expected_in_answer: list[str]
    stale_answer: list[str]


def load_cases(path: Path = DATA_PATH) -> list[MemoryOpsCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    cases: list[MemoryOpsCase] = []
    for item in raw_cases:
        cases.append(
            MemoryOpsCase(
                id=item["id"],
                operation=item["operation"],
                history=list(item["history"]),
                question=item["question"],
                expected_in_answer=list(item["expected_in_answer"]),
                stale_answer=list(item.get("stale_answer", [])),
            )
        )
    return cases


def keyword_match(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def call_answer_llm(prompt: str):
    return chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0,
        timeout=120,
    )


def run_full_context(case: MemoryOpsCase):
    context = "\n".join(f"{i + 1}. {turn}" for i, turn in enumerate(case.history))
    prompt = f"""You answer questions using only the given interaction history.

Rules:
- Use the latest information when facts are updated.
- If the history says a fact should be forgotten, do not reveal it. Say "I don't know" instead.
- Answer briefly.

Interaction history:
{context}

Question: {case.question}
Answer:"""
    response = call_answer_llm(prompt)
    return response, [context]


def run_raw_rag(case: MemoryOpsCase, k: int = 3):
    history_text = "\n".join(case.history)
    chunks = split_text(
        history_text,
        chunk_size=220,
        chunk_overlap=40,
        source_path=f"memory_ops/{case.id}",
    )

    store = LocalVectorStore.build(
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )

    retrieved_records = store.similarity_search(
        case.question,
        k=min(k, len(chunks)),
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    retrieved = [record.page_content for record in retrieved_records]
    context = "\n\n".join(retrieved)

    prompt = f"""You answer questions using only the retrieved memory.

Rules:
- Use the latest information when facts are updated.
- If the retrieved memory says a fact should be forgotten, do not reveal it. Say "I don't know" instead.
- Answer briefly.

Retrieved memory:
{context}

Question: {case.question}
Answer:"""
    response = call_answer_llm(prompt)
    return response, retrieved


def build_summary_memory(case: MemoryOpsCase):
    history = "\n".join(f"{i + 1}. {turn}" for i, turn in enumerate(case.history))
    prompt = f"""Convert the interaction history into compact structured long-term memory.

You must identify:
1. active_facts: current facts that should be remembered.
2. outdated_facts: facts replaced by later updates.
3. forgotten_facts: facts that should not be revealed.
4. short_summary: one concise summary.

Output valid JSON only.

Interaction history:
{history}
"""
    response = chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
        timeout=120,
    )
    return response


def run_summary_memory(case: MemoryOpsCase):
    summary_response = build_summary_memory(case)
    summary_text = summary_response.text

    prompt = f"""You answer questions using structured long-term memory.

Rules:
- Answer using active_facts and short_summary.
- Do not use outdated_facts as the current answer.
- Do not reveal forgotten_facts. If the question asks for forgotten information, say "I don't know."
- Answer briefly.

Structured memory:
{summary_text}

Question: {case.question}
Answer:"""
    answer_response = call_answer_llm(prompt)

    answer_response.prompt_tokens = (
        (answer_response.prompt_tokens or 0) + (summary_response.prompt_tokens or 0)
    )
    answer_response.completion_tokens = (
        (answer_response.completion_tokens or 0) + (summary_response.completion_tokens or 0)
    )
    answer_response.total_tokens = (
        (answer_response.total_tokens or 0) + (summary_response.total_tokens or 0)
    )
    answer_response.latency = answer_response.latency + summary_response.latency

    return answer_response, [summary_text]


def evaluate(method: str, verbose: bool = False):
    cases = load_cases()

    total = len(cases)
    correct = 0
    stale_errors = 0
    total_with_stale = 0
    total_tokens = 0
    total_latency = 0.0

    by_operation = defaultdict(lambda: {"total": 0, "correct": 0, "stale": 0, "with_stale": 0})
    rows = []

    for case in cases:
        start = time.time()
        try:
            if method == "full":
                response, retrieved = run_full_context(case)
            elif method == "rag":
                response, retrieved = run_raw_rag(case)
            elif method == "summary":
                response, retrieved = run_summary_memory(case)
            else:
                raise ValueError(f"Unknown method: {method}")

            answer = response.text.strip()
            tokens = response.total_tokens or 0
            latency = response.latency
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            retrieved = []
            tokens = 0
            latency = time.time() - start

        is_correct = keyword_match(answer, case.expected_in_answer)
        has_stale_error = bool(case.stale_answer) and keyword_match(answer, case.stale_answer)

        if is_correct:
            correct += 1
        if case.stale_answer:
            total_with_stale += 1
        if has_stale_error:
            stale_errors += 1

        total_tokens += tokens
        total_latency += latency

        by_operation[case.operation]["total"] += 1
        by_operation[case.operation]["correct"] += int(is_correct)
        if case.stale_answer:
            by_operation[case.operation]["with_stale"] += 1
        by_operation[case.operation]["stale"] += int(has_stale_error)

        rows.append(
            {
                "id": case.id,
                "operation": case.operation,
                "correct": is_correct,
                "stale_error": has_stale_error,
                "answer": answer,
                "tokens": tokens,
                "latency": latency,
                "retrieved": retrieved,
            }
        )

        if verbose:
            print(f"\n[{case.id}] {case.operation}")
            print(f"Question: {case.question}")
            print(f"Expected: {case.expected_in_answer}")
            print(f"Answer: {answer}")
            print(f"Correct: {is_correct}")
            print(f"Stale error: {has_stale_error}")

    accuracy = correct / total * 100 if total else 0
    stale_error_rate = stale_errors / total_with_stale * 100 if total_with_stale else 0
    avg_tokens = total_tokens / total if total else 0
    avg_latency = total_latency / total if total else 0

    print("\n" + "=" * 80)
    print(f"Memory Operations Evaluation — method={method}")
    print("=" * 80)
    print(f"Chat API         : {describe_chat_target()}")
    print(f"Total cases      : {total}")
    print(f"Accuracy         : {accuracy:.1f}% ({correct}/{total})")
    print(f"Stale Error Rate : {stale_error_rate:.1f}% ({stale_errors}/{total_with_stale})")
    print(f"Avg Tokens       : {avg_tokens:.1f}")
    print(f"Avg Latency      : {avg_latency:.2f}s")

    print("\nBy operation:")
    for operation, stats in sorted(by_operation.items()):
        op_total = stats["total"]
        op_correct = stats["correct"]
        op_acc = op_correct / op_total * 100 if op_total else 0
        op_stale_rate = (
            stats["stale"] / stats["with_stale"] * 100
            if stats["with_stale"]
            else 0
        )
        print(
            f"- {operation:24s} "
            f"acc={op_acc:5.1f}% ({op_correct}/{op_total}), "
            f"stale_error={op_stale_rate:5.1f}%"
        )

    return {
        "method": method,
        "accuracy": accuracy,
        "stale_error_rate": stale_error_rate,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "rows": rows,
        "by_operation": dict(by_operation),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory operations benchmark.")
    parser.add_argument(
        "--method",
        choices=["full", "rag", "summary"],
        required=True,
        help="Which baseline to evaluate.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluate(method=args.method, verbose=args.verbose)


if __name__ == "__main__":
    main()
