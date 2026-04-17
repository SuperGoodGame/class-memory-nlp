import argparse
import os
import shutil

from api_utils import chat_completion, describe_chat_target
from data_utils import TextChunk, load_text, split_text
from embeddings_utils import DEFAULT_EMBEDDING_MODEL
from eval_utils import check_accuracy, check_hit, pad
from query_no_memory import EVAL_DATASET
from vector_store import LocalVectorStore


CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


# ─── Database operations ───────────────────────────────────────────────────
def load_documents(
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    path = os.path.join(DATA_PATH, "alice_in_wonderland.md")
    text = load_text(path)
    return split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source_path=path)


def build_db(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    if os.path.exists(CHROMA_PATH):
        print(f"  Database already exists → {CHROMA_PATH}")
        return

    print("  Loading documents ...")
    chunks = load_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("  Loaded 1 document(s)")

    print(f"  Created {len(chunks)} chunks")

    print("  Building local vector index ...")
    store = LocalVectorStore.build(
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    store.save(CHROMA_PATH)
    print(f"  Saved {store.count()} chunks to {CHROMA_PATH}")


# ─── Retrieval ─────────────────────────────────────────────────────────────
def retrieve(query, k=4):
    store = LocalVectorStore.load(CHROMA_PATH)
    results = store.similarity_search(query, k=k, embedding_model=DEFAULT_EMBEDDING_MODEL)
    return [result.page_content for result in results], results


# ─── LLM call ─────────────────────────────────────────────────────────────
def call_llm(prompt: str):
    return chat_completion(
        [{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
        timeout=60,
    )


# ─── RAG pipeline ───────────────────────────────────────────────────────────
def run_rag(query, k=4):
    retrieved, _ = retrieve(query, k=k)
    context = "\n".join(retrieved)
    full_prompt = f"""Context:
{context}

---
Question: {query}"""
    response = call_llm(full_prompt)
    return response, retrieved


# ─── Evaluation ───────────────────────────────────────────────────────────
def run_evaluation(k=4, verbose=False):
    print("\n" + "=" * 70)
    print("  RAG Evaluation  —  alice_in_wonderland.md")
    print("=" * 70)
    print(f"  Chat API  : {describe_chat_target()}")
    print(f"  Embedding : all-MiniLM-L6-v2")
    print(f"  k         : {k}")
    print("=" * 70 + "\n")

    total = len(EVAL_DATASET)
    hits = 0
    correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_all_tokens = 0
    total_latency = 0.0
    error_messages: list[str] = []
    rows = []

    for i, tc in enumerate(EVAL_DATASET, 1):
        expected_in_docs = getattr(tc, "expected_in_docs", tc.expected_in_answer)
        try:
            response, retrieved = run_rag(tc.query, k=k)
            answer = response.text
            prompt_tokens = response.prompt_tokens or 0
            completion_tokens = response.completion_tokens or 0
            total_tokens = response.total_tokens or 0
            latency = response.latency
        except Exception as e:
            answer = f"[ERROR] {e}"
            retrieved = []
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            latency = 0.0
            error_messages.append(str(e))

        is_hit = check_hit(retrieved, expected_in_docs)
        is_correct = check_accuracy(answer, tc.expected_in_answer)

        if is_hit:
            hits += 1
        if is_correct:
            correct += 1
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_all_tokens += total_tokens
        total_latency += latency

        rows.append({
            "id": i,
            "query": tc.query,
            "hit": "✓" if is_hit else "✗",
            "acc": "✓" if is_correct else "✗",
            "answer": answer,
            "retrieved": retrieved,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        })

        if verbose:
            print(f"[{i}/{total}] {tc.query}")
            print(f"  Expected (docs) : {', '.join(expected_in_docs)}")
            print(f"  Expected (ans)  : {', '.join(tc.expected_in_answer)}")
            for j, doc in enumerate(retrieved, 1):
                print(f"  Doc {j}: {doc[:120]}...")
            print(f"  Answer  : {answer}")
            print(f"  Hit     : {'✓' if is_hit else '✗'}")
            print(f"  Accuracy : {'✓' if is_correct else '✗'}")
            print(f"  Prompt tok: {prompt_tokens}")
            print(f"  Compl tok : {completion_tokens}")
            print(f"  Total tok : {total_tokens}")
            print(f"  Latency   : {latency:.2f}s")
            print()

    hit_rate = hits / total * 100
    accuracy = correct / total * 100
    avg_prompt_tokens = total_prompt_tokens / total if total else 0
    avg_completion_tokens = total_completion_tokens / total if total else 0
    avg_total_tokens = total_all_tokens / total if total else 0
    avg_latency = total_latency / total if total else 0

    qw = 28
    table_width = qw + 8 + 12 + 10 + 3

    print("┌" + "─" * table_width + "┐")
    header = f"│ {'#':^3}  {'Query':<{qw}}  {'Hit':^4}  {'Acc':^4}  {'Tokens':^8}  {'Latency':^8} │"
    print(header)
    print("├" + "─" * table_width + "┤")
    for r in rows:
        latency_text = f"{r['latency']:.2f}s"
        line = (
            f"│ {pad(str(r['id']),3)}  "
            f"{pad(r['query'],qw)}  "
            f"{pad(r['hit'],4)}  "
            f"{pad(r['acc'],4)}  "
            f"{pad(r['total_tokens'],8, '>')}  "
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
        "errors": error_messages,
        "rows": rows,
    }


# ─── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation on alice_in_wonderland.md")
    parser.add_argument("query_text", type=str, nargs="?", help="Single query.")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details.")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks (default: 4).")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the database.")
    args = parser.parse_args()

    if args.rebuild and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"  Removed {CHROMA_PATH}")

    if not os.path.exists(CHROMA_PATH):
        build_db()

    if args.eval:
        run_evaluation(k=args.k, verbose=args.verbose)
    elif args.query_text:
        retrieved, results = retrieve(args.query_text, k=args.k)
        print("\n=== Retrieved Documents ===")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] score={doc.score:.4f}\n{doc.page_content}\n")
        context = "\n".join(retrieved)
        full_prompt = f"""Context:
{context}

---
Question: {args.query_text}"""
        print("=== Final Prompt ===\n" + full_prompt)
        response = call_llm(full_prompt)
        print("\n=== Prompt Length Stats ===")
        print(f"Prompt Tokens     : {response.prompt_tokens}")
        print(f"Completion Tokens : {response.completion_tokens}")
        print(f"Total Tokens      : {response.total_tokens}")
        print(f"Latency           : {response.latency:.2f}s")
        print("\n=== Model Answer ===\n" + response.text)
    else:
        print("Usage:")
        print("  python query_data.py \"question\"     # single query")
        print("  python query_data.py --eval         # run evaluation (table only)")
        print("  python query_data.py --eval --verbose   # evaluation + per-query details")
        print("  python query_data.py --eval --k 6   # retrieve top-k chunks")
        print("  python query_data.py --rebuild      # rebuild database first")


if __name__ == "__main__":
    main()
