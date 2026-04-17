# NLP Baselines

This directory now contains three aligned baselines over `alice_in_wonderland.md`:

- `nlp_baselines/query_no_memory.py`: full-context / no-memory baseline
- `nlp_baselines/query_data.py`: raw chunk retrieval baseline
- `nlp_baselines/query_summary_memory.py`: structured summary-memory baseline

All three use the same remote chat API wrapper in `nlp_baselines/api_utils.py`. Embeddings stay local through `sentence-transformers`, and retrieval uses a small local persisted vector index instead of Chroma.

## Layout

```text
nlp/
├── nlp_baselines/      # core package and runnable baseline modules
├── data/               # source text data
├── class_show/         # report assets and plotting script
├── README.md
├── pixi.toml
└── pixi.lock
```

## Environment

Set one of these API configurations in `.env`:

- `AZURE_OPENAI_API_URL` and `AZURE_OPENAI_API_KEY`
- or `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_API_KEY`
- or `CHAT_API_PROVIDER=dashscope`, `DASHSCOPE_API_KEY`, `CHAT_MODEL=qwen3.6-plus`

Optional:

- `AZURE_OPENAI_API_VERSION`
- `CHAT_API_URL`
- `DASHSCOPE_BASE_URL`

The scripts will fall back to the current HKUST Azure gateway URL if `AZURE_OPENAI_API_URL` is not set.

## Install

```bash
pixi install
```

## Build The Raw Vector Index

```bash
pixi run build-db
```

## Run The Retrieval Baseline

Single query:

```bash
pixi run rag -- "How does Alice meet the Mad Hatter?"
```

Evaluation:

```bash
pixi run rag-eval
```

## Run The Full-Context Baseline

```bash
pixi run full-eval
```

## Run The Summary-Memory Baseline

The first run will generate section summaries through the remote API and cache them under `summary_cache/`.

Single query:

```bash
pixi run summary -- "What does Alice drink that makes her shrink?"
```

Evaluation:

```bash
pixi run summary-eval
```

Force refresh:

```bash
pixi run summary -- --refresh-summaries --rebuild-summary --eval
```

## Notes

- `nlp_baselines/create_database.py` now uses local sentence-transformer embeddings instead of OpenAI embeddings.
- `nlp_baselines/compare_embeddings.py` is also local-only now.
- Retrieval no longer depends on `chromadb`; it uses a lightweight local vector store persisted under `chroma/` and `chroma_summary/`.
- The summary baseline is implemented as `structured summaries + supporting raw chunks`, not summary-only retrieval. This is intentional, because summary-only retrieval tends to lose exact facts and often underperforms raw RAG on detail questions.
