from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


METHODS = ["Full Context", "Raw RAG", "Summary Memory"]
MODELS = ["GPT-4o", "GPT-5.4"]
COLORS = {"GPT-4o": "#E76F51", "GPT-5.4": "#1D4ED8"}

DATA = {
    "GPT-4o": {
        "accuracy": [91.8, 82.0, 85.2],
        "hit_rate": [None, 85.2, 91.8],
        "prompt_tokens": [36709.6, 1228.1, 4587.2],
        "latency": [3.10, 1.97, 2.27],
        "compression_ratio": 0.283,
    },
    "GPT-5.4": {
        "accuracy": [95.1, 91.8, 95.1],
        "hit_rate": [None, 86.9, 98.4],
        "prompt_tokens": [36708.6, 1258.9, 6903.6],
        "latency": [15.89, 14.05, 13.12],
        "compression_ratio": 1.118,
    },
}


def label_bars(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height == 0:
            continue
        ax.annotate(
            f"{height:.1f}",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


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

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.23, hspace=0.3)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_prompt = fig.add_subplot(gs[0, 1])
    ax_latency = fig.add_subplot(gs[1, 0])
    ax_note = fig.add_subplot(gs[1, 1])

    x = np.arange(len(METHODS))
    width = 0.34

    for idx, model in enumerate(MODELS):
        offset = (idx - 0.5) * width
        ax_acc.bar(
            x + offset,
            DATA[model]["accuracy"],
            width=width,
            color=COLORS[model],
            alpha=0.92,
            label=model,
        )
        ax_prompt.bar(
            x + offset,
            DATA[model]["prompt_tokens"],
            width=width,
            color=COLORS[model],
            alpha=0.92,
            label=model,
        )
        ax_latency.bar(
            x + offset,
            DATA[model]["latency"],
            width=width,
            color=COLORS[model],
            alpha=0.92,
            label=model,
        )

    ax_acc.set_title("Accuracy Across Retrieval Baselines")
    ax_acc.set_xticks(x, METHODS)
    ax_acc.set_ylim(78, 100)
    ax_acc.set_ylabel("Accuracy (%)")
    label_bars(ax_acc)
    ax_acc.legend(frameon=False, ncols=2, loc="upper left")

    ax_prompt.set_title("Prompt Token Cost")
    ax_prompt.set_xticks(x, METHODS)
    ax_prompt.set_yscale("log")
    ax_prompt.set_ylabel("Avg Prompt Tokens (log scale)")
    label_bars(ax_prompt)

    ax_latency.set_title("Average Latency")
    ax_latency.set_xticks(x, METHODS)
    ax_latency.set_ylabel("Latency (s)")
    ax_latency.set_ylim(0, 18)
    label_bars(ax_latency)

    ax_note.axis("off")
    ax_note.set_title("Figure Reading")
    note = (
        "Figure 1. Long-context QA result overview.\n\n"
        "The figure summarizes three baselines under two models.\n"
        "Full context still defines the strongest upper bound on quality,\n"
        "but it is by far the most expensive path at roughly 36.7k prompt tokens.\n\n"
        "For GPT-4o, summary memory already improves over raw RAG:\n"
        "52/61 vs 50/61.\n\n"
        "For GPT-5.4, summary memory reaches the same accuracy as full context:\n"
        "58/61, while using far fewer tokens than feeding the full document.\n\n"
        f"Summary compression ratio:\nGPT-4o = {DATA['GPT-4o']['compression_ratio']:.3f}\n"
        f"GPT-5.4 = {DATA['GPT-5.4']['compression_ratio']:.3f}"
    )
    ax_note.text(
        0.0,
        1.0,
        note,
        va="top",
        ha="left",
        fontsize=11,
        color="#1F2937",
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    fig.suptitle(
        "CSIT5520 Result Demo: Long-Context QA Baselines",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "A single overview figure for class-show use. Data source: user-provided GPT-4o metrics and local GPT-5.4 evaluation logs.",
        ha="center",
        fontsize=10,
        color="#57534E",
    )

    png_path = FIG_DIR / "results_overview.png"
    svg_path = FIG_DIR / "results_overview.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
