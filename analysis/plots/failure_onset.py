"""Create the main-text controlled failure-onset figure.

The first two columns show layer-wise GCN trajectories for independently
trained models with total depths D=4 and D=10. The right column reports
final-layer readouts across the complete independently trained depth grid.
The comparison covers high- and low-homophily regimes.
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
RESULTS = os.path.join(ROOT, "results_synthetic_gcn_depth_sweep")

TRAJECTORY_DEPTHS = [4, 10]
DEPTH_GRID = [2, 4, 6, 8, 10, 12, 16, 20]
REGIMES = {
    "high": {
        "label": r"High homophily ($h=0.865$)",
        "color": "#228B22",
        "path": os.path.join(
            RESULTS,
            "gcn_h32_synthetic_high",
            "probe_results.pkl",
        ),
    },
    "low": {
        "label": r"Low homophily ($h=0.110$)",
        "color": "#d62728",
        "path": os.path.join(
            RESULTS,
            "gcn_h32_synthetic_low",
            "probe_results.pkl",
        ),
    },
}


def load_results() -> dict:
    results = {}
    for regime, settings in REGIMES.items():
        with open(settings["path"], "rb") as handle:
            loaded = pickle.load(handle)
        seeds = sorted(loaded)
        if len(seeds) != 5:
            raise ValueError(
                f"Expected five seeds for {regime}; found {seeds}"
            )
        missing = [
            (seed, depth)
            for seed in seeds
            for depth in DEPTH_GRID
            if depth not in loaded[seed]
        ]
        if missing:
            raise ValueError(f"Missing {regime} runs: {missing}")
        results[regime] = loaded
    return results


def trajectory(
    results: dict,
    regime: str,
    depth: int,
    metric: str,
) -> np.ndarray:
    runs = results[regime]
    return np.asarray(
        [
            [record[metric] for record in runs[seed][depth]]
            for seed in sorted(runs)
        ],
        dtype=float,
    )


def final_readout(
    results: dict,
    regime: str,
    metric: str,
) -> np.ndarray:
    runs = results[regime]
    return np.asarray(
        [
            [runs[seed][depth][-1][metric] for depth in DEPTH_GRID]
            for seed in sorted(runs)
        ],
        dtype=float,
    )


def draw_trajectory(
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
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )


def draw_depth_summary(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.errorbar(
        DEPTH_GRID,
        mean,
        yerr=std,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.6,
        markersize=4.0,
        markeredgecolor="white",
        markeredgewidth=0.6,
        capsize=1.8,
        elinewidth=0.8,
        capthick=0.8,
        zorder=3,
    )


def build_figure() -> plt.Figure:
    configure()
    results = load_results()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(5.50, 2.40),
        sharey="row",
        gridspec_kw={
            "width_ratios": [1.0, 1.0, 1.22],
            "hspace": 0.18,
            "wspace": 0.22,
        },
    )

    metric_settings = [
        ("accuracy", "o", "-", r"Accuracy (\%)"),
        ("entropy_normalized", "s", "--", "Normalized entropy"),
    ]
    for column, depth in enumerate(TRAJECTORY_DEPTHS):
        for row, (metric, marker, linestyle, _) in enumerate(metric_settings):
            for regime, settings in REGIMES.items():
                draw_trajectory(
                    axes[row, column],
                    trajectory(results, regime, depth, metric),
                    color=settings["color"],
                    marker=marker,
                    linestyle=linestyle,
                )
            tick_step = 1 if depth == 4 else 2
            axes[row, column].set_xticks(
                np.arange(0, depth + 1, tick_step)
            )

    for row, (metric, marker, linestyle, _) in enumerate(metric_settings):
        settings = REGIMES["high"]
        draw_depth_summary(
            axes[row, 2],
            final_readout(results, "high", metric),
            color=settings["color"],
            marker=marker,
            linestyle=linestyle,
        )

    axes[0, 0].set_title(r"(a) Shallow model, $D=4$", pad=3)
    axes[0, 1].set_title(r"(b) Deep model, $D=10$", pad=3)
    axes[0, 2].set_title(
        "(c) Final readout vs. depth",
        pad=3,
    )

    axes[0, 0].set_ylabel(r"Accuracy (\%)")
    axes[1, 0].set_ylabel("Normalized entropy")
    axes[0, 0].set_ylim(4, 82)
    axes[1, 0].set_ylim(0, 1.02)

    for column in (0, 1):
        axes[1, column].set_xlabel("Message-passing layer")
        axes[0, column].tick_params(labelbottom=False)
    axes[1, 2].set_xlabel(r"Total model depth $D$")
    axes[0, 2].tick_params(labelbottom=False)
    for row in (0, 1):
        axes[row, 2].set_xlim(1.2, 20.8)
        axes[row, 2].set_xticks(DEPTH_GRID)
        axes[row, 2].tick_params(axis="x", labelsize=6.4)

    for ax in axes.flat:
        ax.grid(True)
        ax.tick_params(axis="both", which="major", pad=1.5)

    legend_handles = [
        Line2D(
            [],
            [],
            color=settings["color"],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.6,
            markersize=4.0,
            label=settings["label"],
        )
        for settings in REGIMES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.4,
        handlelength=2.2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.992,
        bottom=0.17,
        top=0.82,
        hspace=0.18,
        wspace=0.22,
    )

    return fig
