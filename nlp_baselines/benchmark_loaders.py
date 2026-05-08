from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LONGBENCH_DATA_DIR = Path("data/longbench")
SUPPORTED_LONGBENCH_DATASETS = ("narrativeqa", "qasper")


@dataclass
class BenchmarkExample:
    dataset: str
    example_id: str
    question: str
    context: str
    answers: list[str]
    length: int | None = None


def load_longbench_sample(dataset: str, *, limit: int | None = None) -> list[BenchmarkExample]:
    if dataset not in SUPPORTED_LONGBENCH_DATASETS:
        raise ValueError(f"Unsupported LongBench dataset: {dataset}")

    path = LONGBENCH_DATA_DIR / f"{dataset}_sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m nlp_baselines.download_longbench_samples` first."
        )

    examples: list[BenchmarkExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(normalize_longbench_record(dataset, raw, fallback_id=str(index)))
            if limit and len(examples) >= limit:
                break
    return examples


def normalize_longbench_record(
    dataset: str,
    raw: dict[str, Any],
    *,
    fallback_id: str,
) -> BenchmarkExample:
    question = str(raw.get("input") or raw.get("question") or "").strip()
    context = str(raw.get("context") or "").strip()
    answers_raw = raw.get("answers", raw.get("answer", []))
    if isinstance(answers_raw, str):
        answers = [answers_raw]
    elif isinstance(answers_raw, list):
        answers = [str(answer).strip() for answer in answers_raw if str(answer).strip()]
    else:
        answers = [str(answers_raw).strip()] if answers_raw is not None else []
    example_id = str(raw.get("_id") or raw.get("id") or fallback_id)
    length_raw = raw.get("length")
    length = int(length_raw) if isinstance(length_raw, int | float | str) and str(length_raw).isdigit() else None
    return BenchmarkExample(
        dataset=dataset,
        example_id=example_id,
        question=question,
        context=context,
        answers=answers,
        length=length,
    )
