from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


METHODS = ["full_context", "raw_rag", "summary_memory"]
METHOD_LABELS = {
    "full_context": "Full Context",
    "raw_rag": "Raw RAG",
    "summary_memory": "Summary Memory",
}
METHOD_SHORT = {
    "full_context": "Full",
    "raw_rag": "RAG",
    "summary_memory": "Summary",
}
MODEL_COLORS = {
    "GPT-4o": "#F97316",
    "GPT-5.4": "#0F766E",
}
METHOD_MARKERS = {
    "full_context": "o",
    "raw_rag": "s",
    "summary_memory": "D",
}


DATA = {
    "GPT-4o": {
        "full_context": {
            "accuracy": 91.8,
            "hit_rate": None,
            "prompt_tokens": 36709.6,
            "total_tokens": 36729.7,
            "latency": 3.10,
            "compression_ratio": None,
        },
        "raw_rag": {
            "accuracy": 82.0,
            "hit_rate": 85.2,
            "prompt_tokens": 1228.1,
            "total_tokens": 1262.6,
            "latency": 1.97,
            "compression_ratio": None,
        },
        "summary_memory": {
            "accuracy": 85.2,
            "hit_rate": 91.8,
            "prompt_tokens": 4587.2,
            "total_tokens": 4616.1,
            "latency": 2.27,
            "compression_ratio": 0.283,
        },
    },
    "GPT-5.4": {
        "full_context": {
            "accuracy": 95.1,
            "hit_rate": None,
            "prompt_tokens": 36708.6,
            "total_tokens": 36725.8,
            "latency": 15.89,
            "compression_ratio": None,
        },
        "raw_rag": {
            "accuracy": 91.8,
            "hit_rate": 86.9,
            "prompt_tokens": 1258.9,
            "total_tokens": 1287.2,
            "latency": 14.05,
            "compression_ratio": None,
        },
        "summary_memory": {
            "accuracy": 95.1,
            "hit_rate": 98.4,
            "prompt_tokens": 6903.6,
            "total_tokens": 6923.7,
            "latency": 13.12,
            "compression_ratio": 1.118,
        },
    },
}


def add_bar_labels(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.1f}",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
        )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFDF8",
            "axes.facecolor": "#FFFDF8",
            "axes.edgecolor": "#D6D3D1",
            "axes.labelcolor": "#1F2937",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 11,
            "grid.color": "#E7E5E4",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "savefig.facecolor": "#FFFDF8",
        }
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    ax_accuracy = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_latency = fig.add_subplot(gs[1, 0])
    ax_hit = fig.add_subplot(gs[1, 1])

    x = np.arange(len(METHODS))
    width = 0.35
    model_names = list(DATA.keys())

    for idx, model in enumerate(model_names):
        offset = (idx - 0.5) * width
        accuracies = [DATA[model][method]["accuracy"] for method in METHODS]
        latencies = [DATA[model][method]["latency"] for method in METHODS]
        ax_accuracy.bar(
            x + offset,
            accuracies,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.92,
        )
        ax_latency.bar(
            x + offset,
            latencies,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.92,
        )

    ax_accuracy.set_title("Accuracy by Retrieval Strategy")
    ax_accuracy.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
    ax_accuracy.set_ylabel("Accuracy (%)")
    ax_accuracy.set_ylim(75, 100)
    add_bar_labels(ax_accuracy)
    ax_accuracy.legend(frameon=False, loc="upper left")

    ax_latency.set_title("Latency Cost by Retrieval Strategy")
    ax_latency.set_xticks(x, [METHOD_LABELS[m] for m in METHODS])
    ax_latency.set_ylabel("Average Latency (s)")
    ax_latency.set_ylim(0, 18)
    add_bar_labels(ax_latency)

    ax_scatter.set_title("Quality vs Prompt Cost")
    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Average Prompt Tokens (log scale)")
    ax_scatter.set_ylabel("Accuracy (%)")
    ax_scatter.set_ylim(80, 97)

    for model in model_names:
        for method in METHODS:
            entry = DATA[model][method]
            ax_scatter.scatter(
                entry["prompt_tokens"],
                entry["accuracy"],
                s=entry["latency"] * 28 + 60,
                color=MODEL_COLORS[model],
                marker=METHOD_MARKERS[method],
                alpha=0.82,
                edgecolors="#111827",
                linewidths=0.8,
            )
            ax_scatter.annotate(
                f"{model}\n{METHOD_SHORT[method]}",
                (entry["prompt_tokens"], entry["accuracy"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                color="#111827",
            )

    ax_scatter.text(
        0.03,
        0.05,
        "Bubble size = average latency",
        transform=ax_scatter.transAxes,
        fontsize=9,
        color="#57534E",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFFBEB", "edgecolor": "#E7E5E4"},
    )

    hit_methods = ["raw_rag", "summary_memory"]
    hit_x = np.arange(len(hit_methods))
    for idx, model in enumerate(model_names):
        offset = (idx - 0.5) * width
        hit_rates = [DATA[model][method]["hit_rate"] for method in hit_methods]
        ax_hit.bar(
            hit_x + offset,
            hit_rates,
            width=width,
            color=MODEL_COLORS[model],
            alpha=0.92,
            label=model,
        )

    ax_hit.set_title("Retrieval Hit Rate and Summary Compression")
    ax_hit.set_xticks(hit_x, [METHOD_LABELS[m] for m in hit_methods])
    ax_hit.set_ylabel("Hit Rate (%)")
    ax_hit.set_ylim(80, 100)
    add_bar_labels(ax_hit)

    compression_text = (
        "Summary compression ratio\n"
        f"GPT-4o: {DATA['GPT-4o']['summary_memory']['compression_ratio']:.3f}\n"
        f"GPT-5.4: {DATA['GPT-5.4']['summary_memory']['compression_ratio']:.3f}"
    )
    ax_hit.text(
        0.03,
        0.10,
        compression_text,
        transform=ax_hit.transAxes,
        fontsize=10,
        color="#292524",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F5F5F4", "edgecolor": "#D6D3D1"},
    )

    fig.suptitle(
        "Class Show Results: GPT-4o vs GPT-5.4 on Long-Context QA Baselines",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Data source: GPT-4o metrics provided by the user; GPT-5.4 metrics from the fluxcode evaluation run.",
        ha="center",
        fontsize=10,
        color="#57534E",
    )

    png_path = FIG_DIR / "gpt4o_vs_gpt54_results.png"
    svg_path = FIG_DIR / "gpt4o_vs_gpt54_results.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
