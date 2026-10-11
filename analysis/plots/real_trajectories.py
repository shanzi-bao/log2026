"""Create real-data accuracy and entropy trajectories for the appendix.

The figure shows Cora and Roman-Empire under GCN and GAT at
independently trained total depths D=4 and D=20. It complements the
controlled results with complete predictive trajectories.
"""

from __future__ import annotations

import os
import pickle

from pathlib import Path

from .style import configure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "results" / "trajectories"
DEPTHS = [4, 20]
DATASETS = {
    "cora": {
        "label": "Cora",
        "color": "#228B22",
    },
    "roman_empire": {
        "label": "Roman-Empire",
        "color": "#d62728",
    },
}
ARCHITECTURES = {
    "gcn": {
        "label": "GCN",
        "prefix": "gcn_h32",
        "marker": "o",
        "linestyle": "-",
    },
    "gat": {
        "label": "GAT",
        "prefix": "sgat4h_relu",
        "marker": "s",
        "linestyle": "--",
    },
}


def load_results(dataset: str, prefix: str) -> dict:
    tuned_path = os.path.join(
        ROOT,
        "results_tuned",
        f"{prefix}_{dataset}",
        "probe_results.pkl",
    )
    deep_path = os.path.join(
        ROOT,
        "results_deep20",
        f"{prefix}_{dataset}",
        "probe_results.pkl",
    )
    with open(tuned_path, "rb") as handle:
        tuned = pickle.load(handle)
    with open(deep_path, "rb") as handle:
        deep = pickle.load(handle)

    seeds = sorted(set(tuned) & set(deep))
    if len(seeds) != 5:
        raise ValueError(
            f"Expected five seeds for {dataset}/{prefix}; got {seeds}"
        )
    results = {}
    for seed in seeds:
        results[seed] = {}
        for depth in DEPTHS:
            source = deep if depth == 20 else tuned
            if depth not in source[seed]:
                raise ValueError(
                    f"Missing depth {depth} for {dataset}/{prefix}/seed {seed}"
                )
            results[seed][depth] = source[seed][depth]
    return results


def trajectory(
    results: dict,
    depth: int,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [record[metric] for record in results[seed][depth]]
            for seed in sorted(results)
        ],
        dtype=float,
    )


def draw(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    layers = np.arange(values.shape[1])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.plot(
        layers,
        mean,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.6,
        markersize=4.0,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )
    ax.fill_between(
        layers,
        mean - std,
        mean + std,
        color=color,
        alpha=0.13,
        linewidth=0,
        zorder=2,
    )


def build_figure() -> plt.Figure:
    configure()
    results = {
        dataset: {
            architecture: load_results(dataset, settings["prefix"])
            for architecture, settings in ARCHITECTURES.items()
        }
        for dataset in DATASETS
    }

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(5.50, 4.15),
        gridspec_kw={"hspace": 0.20, "wspace": 0.20},
    )

    rows = [
        ("cora", "accuracy"),
        ("cora", "entropy_normalized"),
        ("roman_empire", "accuracy"),
        ("roman_empire", "entropy_normalized"),
    ]
    for row, (dataset, metric) in enumerate(rows):
        dataset_settings = DATASETS[dataset]
        for col, depth in enumerate(DEPTHS):
            ax = axes[row, col]
            for architecture, architecture_settings in ARCHITECTURES.items():
                draw(
                    ax,
                    trajectory(
                        results[dataset][architecture],
                        depth,
                        metric,
                    ),
                    color=dataset_settings["color"],
                    marker=architecture_settings["marker"],
                    linestyle=architecture_settings["linestyle"],
                )
            if depth == 4:
                ax.set_xticks(np.arange(0, 5))
            else:
                ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
            ax.grid(True)
            ax.tick_params(axis="both", which="major", pad=1.5)
            if row < 3:
                ax.tick_params(labelbottom=False)
            if col == 1:
                ax.tick_params(labelleft=False)

    axes[0, 0].set_ylim(20, 90)
    axes[0, 0].set_yticks([40, 60, 80])
    axes[0, 1].set_ylim(20, 90)
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_yticks([0, 0.5, 1.0])
    axes[1, 1].set_ylim(0, 1.02)
    axes[2, 0].set_ylim(5, 65)
    axes[2, 0].set_yticks([20, 40, 60])
    axes[2, 1].set_ylim(5, 65)
    axes[3, 0].set_ylim(0.48, 0.94)
    axes[3, 0].set_yticks([0.6, 0.8])
    axes[3, 1].set_ylim(0.48, 0.94)

    axes[0, 0].set_title(r"(a) Total depth $D=4$", pad=3)
    axes[0, 1].set_title(r"(b) Total depth $D=20$", pad=3)
    axes[0, 0].set_ylabel("Cora\n" + r"Accuracy (\%)")
    axes[1, 0].set_ylabel("Cora\nNormalized entropy")
    axes[2, 0].set_ylabel("Roman-Empire\n" + r"Accuracy (\%)")
    axes[3, 0].set_ylabel("Roman-Empire\nNormalized entropy")
    axes[3, 0].set_xlabel("Message-passing layer")
    axes[3, 1].set_xlabel("Message-passing layer")

    legend_handles = [
        Line2D(
            [],
            [],
            color="0.25",
            marker=settings["marker"],
            linestyle=settings["linestyle"],
            linewidth=1.6,
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=settings["label"],
        )
        for settings in ARCHITECTURES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.995,
        bottom=0.10,
        top=0.88,
        hspace=0.20,
        wspace=0.20,
    )

    return fig
