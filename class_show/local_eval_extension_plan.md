# Local Evaluation Extension Plan

This plan is for a follow-up code agent. The goal is to improve the technical report with a broader and more defensible evaluation setup while keeping API cost low.

## Goal

Extend the current AliceQA case study with local-first evaluation. Do not run expensive full LLM answer-generation experiments by default.

The current project already compares:

- Full Context
- Raw RAG
- Structured Summary Memory

The current main QA set has 61 manually curated detail questions from `Alice in Wonderland`. This should be described as a small controlled case study, not a large benchmark.

For the technical report, add at least one standard public benchmark with a local model result. Use LongBench subsets first because they directly match long-context QA.

Recommended standard datasets:

- `LongBench/narrativeqa`: long story QA, closest to the AliceQA setting.
- `LongBench/qasper`: long scientific-paper QA, useful for testing a different document style.
- Optional if time allows: `LongBench/multifieldqa_en`.

Recommended local generation model:

- Gemma 4 local instruction model.

Document the exact model checkpoint used in the README or result file. If Gemma 4 cannot be loaded on the local machine, stop and report the blocker instead of silently switching to a different model.

## Project Structure To Add

Keep new code, data adapters, results, and report notes organized. Do not put everything in the repository root.

Suggested structure:

```text
class-memory-nlp/
├── data/
│   ├── books/
│   │   └── alice_in_wonderland.md
│   └── longbench/
│       ├── narrativeqa_sample.jsonl
│       ├── qasper_sample.jsonl
│       └── multifieldqa_en_sample.jsonl        # optional
├── doc/
│   └── 技术报告design.md
├── nlp_baselines/
│   ├── benchmark_loaders.py                    # LongBench loading / normalization
│   ├── evaluate_retrieval_only.py              # no remote API calls
│   ├── evaluate_local_generation.py            # Gemma 4 local generation
│   ├── local_generation.py                     # local model wrapper
│   ├── query_data.py                           # existing Raw RAG
│   ├── query_summary_memory.py                 # existing Summary Memory
│   └── ...
├── results/
│   ├── aliceqa_retrieval_only.json
│   ├── longbench_retrieval_only.json
│   ├── local_generation_gemma4.json
│   └── tables/
│       ├── retrieval_summary.csv
│       └── generation_summary.csv
└── .cache/
    ├── huggingface/
    └── models/
```

Cache rule:

- Keep Hugging Face datasets/models under repo-local `.cache/`.
- Set `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and related cache env vars to paths inside `.cache/` before downloading or loading models.
- Do not write large downloads to the user's global home cache.

## Main Principle

Separate retrieval evaluation from generation evaluation.

Structured Summary Memory mainly changes how context is represented and retrieved. Therefore, we can evaluate most of the method locally by checking whether retrieved context contains answer-relevant evidence.

## What To Implement

### 1. Add Retrieval-Only Evaluation

Create a new module:

```text
nlp_baselines/evaluate_retrieval_only.py
```

It should evaluate Raw RAG and Summary Memory without calling the chat API for final answer generation.

For each of the 61 `EVAL_DATASET` questions:

- Retrieve top-k raw chunks for Raw RAG.
- Retrieve top-k summary memories for Summary Memory.
- Collect supporting raw chunks for Summary Memory using the existing logic.
- Check whether retrieved text contains the expected answer keywords.

Suggested metrics:

- `hit_rate`
- `hits / total`
- `avg_retrieved_chars`
- `avg_retrieved_records`
- `retrieval_latency`
- optional: `Recall@1`, `Recall@3`, `Recall@5` if easy to support

Use existing utilities where possible:

- `EVAL_DATASET` from `nlp_baselines/query_no_memory.py`
- `check_hit` from `nlp_baselines/eval_utils.py`
- Raw RAG retrieval from `nlp_baselines/query_data.py`
- Summary retrieval from `nlp_baselines/query_summary_memory.py`

Important: this script should not call `chat_completion()` during normal retrieval-only evaluation.

### 2. Add Cost / Context-Size Analysis

In the same module, report approximate context size for each strategy:

- Full Context: full document length per query.
- Raw RAG: sum of retrieved raw chunk lengths.
- Summary Memory: sum of retrieved summary memory lengths plus supporting raw chunk lengths.

Report both:

- character length
- approximate token length if a simple tokenizer is available

If no tokenizer is available, use a simple approximation and label it clearly as approximate.

Suggested output:

```text
Strategy              Hit Rate    Avg Context Chars    Avg Retrieval Latency
Raw RAG               ...
Summary Memory         ...
Full Context           N/A         ...
```

Full Context does not need retrieval hit rate because it contains the whole document.

### 3. Add Optional Local Generation Sanity Check

This part is required for the technical report, but should be sample-limited so it stays cheap and runnable.

Add a flag such as:

```bash
python -m nlp_baselines.evaluate_local_generation --dataset narrativeqa --limit 50
```

Use Gemma 4 locally for answer generation. The same local model should be used for Raw RAG and Summary Memory so the comparison is fair.

Start with small samples:

- AliceQA: all 61 questions.
- NarrativeQA: 50 examples.
- Qasper: 50 examples.
- Optional MultiFieldQA-en: 50 examples.

If Gemma 4 setup is not available, print a clear message and stop. Do not call GPT-4o/GPT-5.4 as a fallback.

This result should be described as a local-model benchmark, not a high-end LLM benchmark.

### 4. Add Optional Broader Benchmark Notes

Add a short Markdown note:

```text
class_show/benchmark_extension_notes.md
```

It should explain that the current 61-question AliceQA set is a small controlled case study. For a broader evaluation, suggest:

- LongBench: `narrativeqa`, `qasper`, `multifieldqa_en`
- NarrativeQA for long story QA
- Qasper for long scientific-paper QA
- LongMemEval or LoCoMo for memory-specific behavior

For this assignment, implement at least a small sampled LongBench experiment. The note should still explain why these datasets were selected and what remains future work.

## Suggested Commands

Add pixi tasks if possible:

```toml
retrieval-eval = "python -m nlp_baselines.evaluate_retrieval_only"
retrieval-eval-verbose = "python -m nlp_baselines.evaluate_retrieval_only --verbose"
local-gemma4-eval = "python -m nlp_baselines.evaluate_local_generation --model gemma4 --dataset narrativeqa --limit 50"
```

If not using pixi locally, make sure this also works:

```bash
python -m nlp_baselines.evaluate_retrieval_only
```

## Reporting Language

Use this framing in the report:

> Our current AliceQA evaluation is a small controlled case study rather than a large benchmark. To reduce API cost, we separate retrieval evaluation from final answer generation. Since our method mainly changes the memory representation and retrieval stage, retrieval-only evaluation checks whether the retrieved context contains answer-relevant evidence without calling a large remote model.

And:

> Summary Memory has a one-time preprocessing cost because each section is summarized once. After the memory is built and cached, all questions reuse the same memory index. This reduces the active context used per query, even though full-context prompting could also benefit from prompt caching or KV cache.

## Acceptance Criteria

The implementation is acceptable if:

- Retrieval-only evaluation runs without remote chat API calls.
- It evaluates all 61 existing questions.
- It compares Raw RAG and Summary Memory retrieval hit rate.
- It reports average context size.
- It does not require expensive API calls by default.
- It runs at least one standard LongBench subset with Gemma 4 local generation.
- It saves machine-readable results under `results/`.
- It keeps code under `nlp_baselines/`, report design under `doc/`, data samples under `data/longbench/`, and generated tables under `results/tables/`.
- It writes any new cache or generated artifacts inside the repository directory.
- It updates README or adds clear notes explaining how to run the local evaluation.

## Do Not Do

- Do not claim this is a large benchmark.
- Do not run full GPT-4o/GPT-5.4 answer generation unless explicitly requested.
- Do not download large model files outside the repository cache.
- Do not remove existing presentation or report files.
- Do not rewrite the whole project structure unless necessary.
- Do not silently replace Gemma 4 with another model. If Gemma 4 cannot run locally, report the blocker.
