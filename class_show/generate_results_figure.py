from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["Full Context", "Raw RAG", "Summary Memory"]
STRATEGY_COLORS = {
    "Full Context": "#1D4ED8",
    "Raw RAG": "#F97316",
    "Summary Memory": "#0F766E",
}
MODEL_THEMES = {
    "GPT-4o": {"accent": "#B45309", "filename": "gpt4o_memory_strategies"},
    "GPT-5.4": {"accent": "#1E3A8A", "filename": "gpt54_memory_strategies"},
    "Gemma4 Local": {"accent": "#0F766E", "filename": "gemma4_memory_strategies"},
}
LONGBENCH_DATASETS = ["narrativeqa", "qasper"]
LONGBENCH_METHODS = ["Raw RAG", "Summary Memory (structured)"]
LONGBENCH_RESULT_FILES = {
    "retrieval": ROOT.parent / "results" / "longbench_retrieval_only.json",
    "generation": ROOT.parent / "results" / "local_generation_gemma4_longbench.json",
}

DATA = {
    "GPT-4o": {
        "accuracy": [91.8, 82.0, 85.2],
        "hit_rate": [np.nan, 85.2, 91.8],
        "prompt_tokens": [36709.6, 1228.1, 4587.2],
        "latency": [3.10, 1.97, 2.27],
        "compression_ratio": 0.283,
        "summary_note": (
            "Summary memory improves over raw RAG by 2 correct answers\n"
            "(52/61 vs 50/61) while remaining much cheaper than full context."
        ),
    },
    "GPT-5.4": {
        "accuracy": [95.1, 91.8, 95.1],
        "hit_rate": [np.nan, 86.9, 98.4],
        "prompt_tokens": [36708.6, 1258.9, 6903.6],
        "latency": [15.89, 14.05, 13.12],
        "compression_ratio": 1.118,
        "summary_note": (
            "Summary memory matches full context accuracy at 58/61,\n"
            "while cutting prompt tokens from 36.7k to 6.9k."
        ),
    },
    "Gemma4 Local": {
        "accuracy": [np.nan, 72.1, 90.2],
        "hit_rate": [np.nan, 86.9, 90.2],
        "prompt_tokens": [35983.0, 1369.4, 1959.9],
        "latency": [np.nan, 14.25, 39.74],
        "compression_ratio": 1.118,
        "summary_note": (
            "Local Gemma4 cannot run full context because the Alice document\n"
            "exceeds the effective llama.cpp slot context. Summary memory\n"
            "improves local accuracy from 44/61 to 55/61 over raw RAG."
        ),
    },
}


def label_bars(ax: plt.Axes, decimals: int = 1) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height == 0:
            continue
        ax.annotate(
            f"{height:.{decimals}f}",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


def build_model_figure(model_name: str, index: int) -> None:
    theme = MODEL_THEMES[model_name]
    metrics = DATA[model_name]
    x = np.arange(len(STRATEGIES))

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    ax_accuracy = fig.add_subplot(gs[0, 0])
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_hit = fig.add_subplot(gs[1, 0])
    ax_note = fig.add_subplot(gs[1, 1])

    bar_colors = [STRATEGY_COLORS[s] for s in STRATEGIES]

    ax_accuracy.bar(x, metrics["accuracy"], color=bar_colors, width=0.62)
    ax_accuracy.set_title("Accuracy by Memory Strategy")
    ax_accuracy.set_xticks(x, STRATEGIES)
    ax_accuracy.set_ylim(65, 100)
    ax_accuracy.set_ylabel("Accuracy (%)")
    label_bars(ax_accuracy)

    ax_cost.bar(x, metrics["prompt_tokens"], color=bar_colors, width=0.62)
    ax_cost.set_title("Prompt Token Cost")
    ax_cost.set_xticks(x, STRATEGIES)
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("Average Prompt Tokens (log scale)")
    label_bars(ax_cost)

    hit_x = np.arange(2)
    hit_values = [metrics["hit_rate"][1], metrics["hit_rate"][2]]
    hit_labels = ["Raw RAG", "Summary Memory"]
    hit_colors = [STRATEGY_COLORS["Raw RAG"], STRATEGY_COLORS["Summary Memory"]]
    ax_hit.bar(hit_x, hit_values, color=hit_colors, width=0.56)
    ax_hit.set_title("Retrieval Hit Rate and Latency")
    ax_hit.set_xticks(hit_x, hit_labels)
    ax_hit.set_ylim(80, 100)
    ax_hit.set_ylabel("Hit Rate (%)")
    label_bars(ax_hit)

    full_latency = "N/A" if not np.isfinite(metrics["latency"][0]) else f"{metrics['latency'][0]:.2f}s"
    latency_text = (
        "Average latency\n"
        f"Full Context: {full_latency}\n"
        f"Raw RAG: {metrics['latency'][1]:.2f}s\n"
        f"Summary: {metrics['latency'][2]:.2f}s"
    )
    ax_hit.text(
        1.05,
        0.15,
        latency_text,
        transform=ax_hit.transAxes,
        fontsize=10,
        color="#1F2937",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    ax_note.axis("off")
    ax_note.set_title("Result Interpretation")
    note_text = (
        f"Figure {index}. {model_name} memory-strategy comparison.\n\n"
        f"{metrics['summary_note']}\n\n"
        "Reading guide:\n"
        "- Full context is the strongest upper bound, but also the most expensive.\n"
        "- Raw RAG is the cheapest baseline.\n"
        "- Summary memory is the middle point on cost, and the more promising memory strategy.\n\n"
        f"Summary compression ratio: {metrics['compression_ratio']:.3f}"
    )
    if model_name == "GPT-5.4":
        note_text += "\nOne evaluation sample timed out during the run."
    if model_name == "Gemma4 Local":
        note_text += "\nFull context is shown as context-cost reference, not a local run."
    ax_note.text(
        0.0,
        1.0,
        note_text,
        va="top",
        ha="left",
        fontsize=11,
        color="#1F2937",
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#FFFBEB", "edgecolor": "#FCD34D"},
    )

    fig.suptitle(
        f"CSIT5520 Result Demo: {model_name} Memory Strategy Comparison",
        fontsize=18,
        fontweight="bold",
        color=theme["accent"],
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Theme: compare memory strategies rather than compare models directly.",
        ha="center",
        fontsize=10,
        color="#57534E",
    )

    png_path = FIG_DIR / f"{theme['filename']}.png"
    svg_path = FIG_DIR / f"{theme['filename']}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def load_structured_longbench_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("memory_mode") != "structured":
        return None
    return {
        (str(row["dataset"]), str(row["method"])): row
        for row in payload.get("summary", [])
    }


def build_longbench_figure(index: int) -> None:
    retrieval = load_structured_longbench_summary(LONGBENCH_RESULT_FILES["retrieval"])
    generation = load_structured_longbench_summary(LONGBENCH_RESULT_FILES["generation"])
    if retrieval is None or generation is None:
        print("Skipped LongBench figure because structured result files are not ready.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    x = np.arange(len(LONGBENCH_DATASETS))
    width = 0.34
    method_colors = [
        STRATEGY_COLORS["Raw RAG"],
        STRATEGY_COLORS["Summary Memory"],
    ]

    panels = [
        (axes[0], retrieval, "Retrieval Hit Rate", "hit"),
        (axes[1], generation, "Answer Accuracy", "accuracy"),
    ]
    for ax, summary, title, score_name in panels:
        for offset, method in zip([-width / 2, width / 2], LONGBENCH_METHODS):
            values = [
                float(summary.get((dataset, method), {}).get("score", 0.0))
                for dataset in LONGBENCH_DATASETS
            ]
            label = "Summary Memory" if method.startswith("Summary Memory") else method
            color = method_colors[1] if method.startswith("Summary Memory") else method_colors[0]
            ax.bar(x + offset, values, width=width, label=label, color=color)
        ax.set_title(title)
        ax.set_xticks(x, [dataset.upper() for dataset in LONGBENCH_DATASETS])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        label_bars(ax)
        ax.grid(axis="y", alpha=0.6)
        for dataset_index, dataset in enumerate(LONGBENCH_DATASETS):
            raw_total = summary.get((dataset, "Raw RAG"), {}).get("total", 0)
            ax.text(
                dataset_index,
                -0.12,
                f"n={raw_total}",
                ha="center",
                va="top",
                fontsize=9,
                color="#57534E",
                transform=ax.get_xaxis_transform(),
            )
        ax.text(
            0.01,
            0.98,
            f"Metric: {score_name}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="#44403C",
        )

    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle(
        f"Figure {index}. Gemma4 Local on LongBench Structured Memory",
        fontsize=18,
        fontweight="bold",
        color=MODEL_THEMES["Gemma4 Local"]["accent"],
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "LongBench sampled subsets use official THUDM examples; Summary Memory uses Gemma4-generated JSON records plus raw evidence.",
        ha="center",
        fontsize=10,
        color="#57534E",
    )
    png_path = FIG_DIR / "longbench_gemma4_structured_results.png"
    svg_path = FIG_DIR / "longbench_gemma4_structured_results.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFDF7",
            "axes.facecolor": "#FFFDF7",
            "savefig.facecolor": "#FFFDF7",
            "axes.edgecolor": "#D6D3D1",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 11,
            "grid.color": "#E7E5E4",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )
    build_model_figure("GPT-4o", 1)
    build_model_figure("GPT-5.4", 2)
    build_model_figure("Gemma4 Local", 3)
    build_longbench_figure(4)


if __name__ == "__main__":
    main()
