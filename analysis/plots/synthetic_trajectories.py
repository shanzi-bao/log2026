"""Create controlled failure-onset trajectory figures.

Each architecture is shown as a 2 x 3 grid:
  - columns: independently trained total depths D=4, 10, and 20;
  - rows: layer-wise probe accuracy and normalized predictive entropy;
  - curves: high-, intermediate-, and low-homophily controlled graphs.

Curves show means and shaded regions show one standard deviation over five
seeds.
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
DEPTHS = [4, 10, 20]

SETTINGS = {
    "group": "results_synthetic_gat_extension",
    "prefix": "sgat4h_relu",
}

REGIMES = {
    "high": {
        "label": r"High homophily ($h=0.865$)",
        "color": "#228B22",
    },
    "mid": {
        "label": r"Mid homophily ($h=0.512$)",
        "color": "#ff9800",
    },
    "low": {
        "label": r"Low homophily ($h=0.110$)",
        "color": "#d62728",
    },
}


def result_path(settings: dict, regime: str) -> str:
    directory = f"{settings['prefix']}_synthetic_{regime}"
    return os.path.join(
        ROOT,
        settings["group"],
        directory,
        "probe_results.pkl",
    )


def load_results(settings: dict) -> dict:
    results = {}
    for regime in REGIMES:
        path = result_path(settings, regime)
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
        seeds = sorted(loaded)
        if len(seeds) != 5:
            raise ValueError(f"Expected five seeds in {path}; got {seeds}")
        missing = [
            (seed, depth)
            for seed in seeds
            for depth in DEPTHS
            if depth not in loaded[seed]
        ]
        if missing:
            raise ValueError(f"Missing runs in {path}: {missing}")
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
        linewidth=1.45,
        markersize=3.2,
        zorder=3,
    )
    ax.fill_between(
        layers,
        mean - std,
        mean + std,
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    ax.grid(alpha=0.19, linewidth=0.6)


def build_figure() -> plt.Figure:
    configure()
    results = load_results(SETTINGS)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.10, 3.20),
        sharey="row",
        gridspec_kw={"hspace": 0.18, "wspace": 0.13},
    )

    for column, depth in enumerate(DEPTHS):
        for regime, regime_settings in REGIMES.items():
            draw(
                axes[0, column],
                trajectory(results, regime, depth, "accuracy"),
                color=regime_settings["color"],
                marker="o",
                linestyle="-",
            )
            draw(
                axes[1, column],
                trajectory(
                    results,
                    regime,
                    depth,
                    "entropy_normalized",
                ),
                color=regime_settings["color"],
                marker="s",
                linestyle="--",
            )

        axes[0, column].set_title(
            f"Total depth $D={depth}$",
            fontsize=8.0,
            pad=3,
        )
        tick_step = 1 if depth <= 4 else 2 if depth <= 10 else 4
        ticks = np.arange(0, depth + 1, tick_step)
        axes[0, column].set_xticks(ticks)
        axes[1, column].set_xticks(ticks)
        axes[0, column].tick_params(labelbottom=False)

    axes[0, 0].set_ylabel(r"Accuracy (\%)", fontsize=7.4)
    axes[1, 0].set_ylabel("Normalized entropy", fontsize=7.4)
    axes[0, 0].set_ylim(4, 82)
    axes[1, 0].set_ylim(0, 1.02)
    for ax in axes.flat:
        ax.tick_params(labelsize=6.8)

    legend = [
        Line2D(
            [],
            [],
            color=settings["color"],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=settings["label"],
        )
        for settings in REGIMES.values()
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=7.0,
        columnspacing=1.0,
        handlelength=1.9,
    )
    fig.supxlabel("Message-passing layer", fontsize=7.4, y=0.012)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.17,
        top=0.82,
        hspace=0.18,
        wspace=0.13,
    )

    return fig
