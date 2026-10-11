"""Create a compact entropy-focused summary of controlled failure onset.

The figure combines the independently trained GCN and GAT depth sweeps:

  (a) first-step normalized-entropy change across total model depths;
  (b) final normalized entropy under high homophily;
  (c) final probe accuracy under high homophily.

Curves show the mean across five seeds and error bars show one standard
deviation.
The complete layer-wise trajectory grids remain separate appendix figures.
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

ARCHITECTURES = {
    "GCN": {
        "depths": [2, 4, 6, 8, 10, 12, 16, 20],
        "marker": "o",
        "linestyle": "-",
        "paths": {
            "high": os.path.join(
                ROOT,
                "results_synthetic_gcn_depth_sweep",
                "gcn_h32_synthetic_high",
                "probe_results.pkl",
            ),
            "low": os.path.join(
                ROOT,
                "results_synthetic_gcn_depth_sweep",
                "gcn_h32_synthetic_low",
                "probe_results.pkl",
            ),
        },
    },
    "GAT": {
        "depths": [2, 4, 6, 8, 10, 12, 16, 20],
        "marker": "s",
        "linestyle": "--",
        "paths": {
            "high": os.path.join(
                ROOT,
                "results_synthetic_gat_extension",
                "sgat4h_relu_synthetic_high",
                "probe_results.pkl",
            ),
            "low": os.path.join(
                ROOT,
                "results_synthetic_gat_extension",
                "sgat4h_relu_synthetic_low",
                "probe_results.pkl",
            ),
        },
    },
}

REGIMES = {
    "high": {
        "label": r"High homophily ($h=0.865$)",
        "color": "#228B22",
    },
    "low": {
        "label": r"Low homophily ($h=0.110$)",
        "color": "#d62728",
    },
}


def load_results() -> dict:
    results = {}
    for architecture, settings in ARCHITECTURES.items():
        results[architecture] = {}
        for regime, path in settings["paths"].items():
            with open(path, "rb") as handle:
                loaded = pickle.load(handle)
            missing = [
                (seed, depth)
                for seed in sorted(loaded)
                for depth in settings["depths"]
                if depth not in loaded[seed]
            ]
            if missing:
                raise ValueError(
                    f"Missing {architecture}/{regime} runs: {missing}"
                )
            results[architecture][regime] = loaded
    return results


def seed_values(
    results: dict,
    architecture: str,
    regime: str,
    depth: int,
    summary: str,
) -> np.ndarray:
    runs = results[architecture][regime]
    values = []
    for seed in sorted(runs):
        records = runs[seed][depth]
        if summary == "first_hop_entropy":
            value = (
                records[1]["entropy_normalized"]
                - records[0]["entropy_normalized"]
            )
        elif summary == "final_accuracy":
            value = records[-1]["accuracy"]
        elif summary == "final_entropy":
            value = records[-1]["entropy_normalized"]
        else:
            raise KeyError(summary)
        values.append(value)
    return np.asarray(values, dtype=float)


def mean_and_std(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std())


def plot_series(
    ax: plt.Axes,
    results: dict,
    architecture: str,
    regime: str,
    summary: str,
) -> None:
    settings = ARCHITECTURES[architecture]
    means = []
    stds = []
    for depth in settings["depths"]:
        mean, std = mean_and_std(
            seed_values(
                results,
                architecture,
                regime,
                depth,
                summary,
            )
        )
        means.append(mean)
        stds.append(std)

    ax.errorbar(
        settings["depths"],
        means,
        yerr=stds,
        color=REGIMES[regime]["color"],
        marker=settings["marker"],
        linestyle=settings["linestyle"],
        linewidth=1.35,
        markersize=3.6,
        capsize=1.8,
        elinewidth=0.8,
        capthick=0.8,
        zorder=3,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(1, 21)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 16, 20])
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=7.1)


def build_figure() -> plt.Figure:
    configure()
    results = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(5.50, 1.82), sharex=True)

    for architecture in ARCHITECTURES:
        for regime in REGIMES:
            plot_series(
                axes[0],
                results,
                architecture,
                regime,
                "first_hop_entropy",
            )

        plot_series(
            axes[1],
            results,
            architecture,
            "high",
            "final_entropy",
        )
        plot_series(
            axes[2],
            results,
            architecture,
            "high",
            "final_accuracy",
        )

    axes[0].axhline(0, color="0.35", linewidth=0.8, linestyle=":")

    axes[0].set_title("(a) First-step entropy", fontsize=8.7)
    axes[1].set_title(
        r"(b) Final entropy ($h=0.865$)", fontsize=8.7
    )
    axes[2].set_title(
        r"(c) Final accuracy ($h=0.865$)", fontsize=8.7
    )

    axes[0].set_ylabel(
        r"$\Delta \overline{H}_{0\rightarrow1}$",
        fontsize=7.8,
    )
    axes[1].set_ylabel("Final normalized entropy", fontsize=7.8)
    axes[2].set_ylabel(r"Final probe accuracy (\%)", fontsize=7.8)

    axes[0].set_ylim(-0.34, 0.43)
    axes[1].set_ylim(0, 0.70)
    axes[2].set_ylim(45, 79)

    for ax in axes:
        style_axis(ax)
        ax.set_xlabel(r"Total model depth $D$", fontsize=7.8)

    legend_handles = [
        Line2D(
            [],
            [],
            color=settings["color"],
            linewidth=2,
            label=settings["label"],
        )
        for settings in REGIMES.values()
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                color="0.20",
                marker=settings["marker"],
                linestyle=settings["linestyle"],
                linewidth=1.4,
                markersize=4.0,
                label=architecture,
            )
            for architecture, settings in ARCHITECTURES.items()
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=7.5,
        columnspacing=1.25,
        handlelength=2.2,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.22,
        top=0.72,
        wspace=0.38,
    )

    return fig
