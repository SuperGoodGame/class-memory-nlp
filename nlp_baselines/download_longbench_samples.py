from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests

from .benchmark_loaders import LONGBENCH_DATA_DIR, SUPPORTED_LONGBENCH_DATASETS


LONGBENCH_DATA_ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
SOURCE_DIR = LONGBENCH_DATA_DIR / "source"
ZIP_PATH = SOURCE_DIR / "data.zip"


def download_zip(*, force: bool = False) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        return ZIP_PATH

    session = requests.Session()
    session.trust_env = False
    with session.get(LONGBENCH_DATA_ZIP_URL, stream=True, timeout=300) as response:
        if not response.ok:
            raise RuntimeError(
                f"Failed to download LongBench data.zip: {response.status_code} {response.text[:300]}"
            )
        tmp_path = ZIP_PATH.with_suffix(".zip.part")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(ZIP_PATH)
    return ZIP_PATH


def find_member(zip_file: zipfile.ZipFile, dataset: str) -> str:
    candidates = [
        name
        for name in zip_file.namelist()
        if name.endswith(f"{dataset}.jsonl") and not name.endswith("/")
    ]
    if not candidates:
        raise RuntimeError(f"Could not find {dataset}.jsonl in {ZIP_PATH}")
    return sorted(candidates, key=len)[0]


def write_sample(dataset: str, *, sample_size: int, force: bool = False) -> Path:
    output_path = LONGBENCH_DATA_DIR / f"{dataset}_sample.jsonl"
    if output_path.exists() and not force:
        return output_path

    with zipfile.ZipFile(ZIP_PATH) as archive:
        member = find_member(archive, dataset)
        rows = []
        with archive.open(member) as raw_handle:
            for line in raw_handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line.decode("utf-8")))
                if len(rows) >= sample_size:
                    break

    LONGBENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sampled LongBench datasets into data/longbench.")
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_LONGBENCH_DATASETS))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_zip(force=args.force)
    for dataset in args.datasets:
        if dataset not in SUPPORTED_LONGBENCH_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        path = write_sample(dataset, sample_size=args.sample_size, force=args.force)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
