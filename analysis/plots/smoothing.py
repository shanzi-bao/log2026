"""Predictive consequences under geometric smoothing.

The main-text figure compares beneficial, neutral, and harmful smoothing on
Cora, Amazon-Ratings, and Roman-Empire, respectively.
"""
from __future__ import annotations

import os
import pickle

from pathlib import Path

from .style import configure
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "results" / "trajectories"
PKL = os.path.join(
    ROOT,
    "results_geometry_grid_paired_splits",
    "geometry_grid_results.pkl",
)
GLOBAL_MAD_PKL = os.path.join(
    ROOT,
    "results_geometry_global_paired_full",
    "real_depth4_canonical_geometry.pkl",
)

MAIN_DATASETS = [
    ("cora", "(a) Cora", "Beneficial smoothing", 0.77),
    (
        "amazon-ratings",
        "(b) Amazon-Ratings",
        "Neutral smoothing",
        0.14,
    ),
    (
        "roman-empire",
        "(c) Roman-Empire",
        "Harmful smoothing",
        -0.05,
    ),
]

LAYERS = np.arange(5)


def probe_trajectories(results: dict, dataset: str, key: str) -> np.ndarray:
    models = results[dataset]["models"]["gcn"]
    return np.asarray(
        [[record[key] for record in models[seed]["probe"]] for seed in sorted(models)],
        dtype=float,
    )


def representation_trajectories(
    results: dict, dataset: str, metric: str
) -> np.ndarray:
    models = results[dataset]["models"]["gcn"]
    trajectories = []
    for seed in sorted(models):
        geometry = models[seed]["geometry"]
        if metric not in {"dirichlet", "mad_global"}:
            raise KeyError(metric)
        values = np.asarray([record[metric] for record in geometry], dtype=float)
        trajectories.append(values / np.maximum(values[0], 1e-30))
    return np.asarray(trajectories, dtype=float)


def draw_mean_band(ax, values: np.ndarray, *, color: str, marker: str, label: str,
                   linestyle: str = "-") -> None:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.fill_between(
        LAYERS,
        mean - std,
        mean + std,
        color=color,
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    ax.errorbar(
        LAYERS,
        mean,
        yerr=std,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.6,
        markersize=3.3,
        markeredgecolor="white",
        markeredgewidth=0.5,
        ecolor=color,
        elinewidth=0.7,
        capsize=1.8,
        capthick=0.7,
        label=label,
        zorder=3,
    )


def endpoint_annotation(
    ax,
    values: np.ndarray,
    *,
    accuracy: bool,
    x: float = 0.03,
    y: float = 0.015,
    ha: str = "left",
    va: str = "bottom",
) -> None:
    delta = float(values.mean(axis=0)[-1] - values.mean(axis=0)[0])
    suffix = " pp" if accuracy else ""
    formatted_delta = f"{delta:+.1f}" if accuracy else f"{delta:+.2f}"
    ax.text(
        x,
        y,
        rf"$\Delta_{{0\to4}}={formatted_delta}${suffix}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 1.4},
    )


def render_main(results: dict, datasets: list[tuple]) -> plt.Figure:
    """Render datasets as rows and metrics as columns for a compact main figure."""
    row_labels = []
    fig, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(5.50, 2.85),
        sharex=True,
        sharey="col",
        gridspec_kw={"hspace": 0.13, "wspace": 0.20},
        squeeze=False,
    )

    for row, (
        dataset,
        panel_title,
        regime_title,
        adjusted_homophily,
    ) in enumerate(datasets):
        accuracy = probe_trajectories(results, dataset, "accuracy")
        entropy = probe_trajectories(results, dataset, "entropy_normalized")

        draw_mean_band(
            axes[row, 0],
            accuracy,
            color="#1f77b4",
            marker="o",
            label="Probe accuracy",
        )
        endpoint_annotation(axes[row, 0], accuracy, accuracy=True)

        draw_mean_band(
            axes[row, 1],
            entropy,
            color="#d62728",
            marker="s",
            linestyle="--",
            label="Normalized entropy",
        )
        endpoint_annotation(axes[row, 1], entropy, accuracy=False)

        metric_styles = [
            ("dirichlet", "DE (scale-normalized)", "0.35", "o", "-"),
            ("mad_global", "MAD (global)", "0.60", "s", "--"),
        ]
        for key, label, color, marker, linestyle in metric_styles:
            draw_mean_band(
                axes[row, 2],
                representation_trajectories(results, dataset, key),
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )

        row_labels.append(
            (
                panel_title,
                rf"$h_{{\mathrm{{adj}}}}={adjusted_homophily:.2f}$",
                regime_title,
            )
        )

    axes[0, 0].set_title(r"Probe accuracy (\%)", pad=3)
    axes[0, 1].set_title("Normalized entropy", pad=3)
    axes[0, 2].set_title("Input-normalized geometry", pad=3)
    axes[0, 0].set_ylim(0, 82)
    axes[0, 0].set_yticks([25, 50, 75])
    axes[0, 1].set_ylim(-0.20, 0.95)
    axes[0, 1].set_yticks([0.00, 0.25, 0.50, 0.75])
    axes[0, 2].set_ylim(-0.04, 1.06)

    for ax in axes.flat:
        ax.set_xticks(LAYERS)
        ax.grid(True)
        ax.tick_params(labelsize=7)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    axes[0, 2].legend(
        handles,
        labels,
        loc="upper right",
        ncol=1,
        fontsize=6.3,
        handlelength=1.6,
        handletextpad=0.45,
        borderaxespad=0.15,
        labelspacing=0.10,
        frameon=False,
    )
    fig.supxlabel("Layer", fontsize=8, y=0.012)
    fig.subplots_adjust(
        left=0.23,
        right=0.975,
        bottom=0.14,
        top=0.89,
        hspace=0.13,
        wspace=0.20,
    )
    for row, (dataset_label, homophily_label, regime_label) in enumerate(row_labels):
        position = axes[row, 0].get_position()
        center_y = (position.y0 + position.y1) / 2
        fig.text(
            0.012,
            center_y + 0.042,
            dataset_label,
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        fig.text(
            0.012,
            center_y,
            homophily_label,
            ha="left",
            va="center",
            fontsize=7,
            color="0.4",
        )
        fig.text(
            0.012,
            center_y - 0.042,
            regime_label,
            ha="left",
            va="center",
            fontsize=7,
            color="0.4",
        )
    return fig


def load_results(results_path=None) -> dict:
    """Read a combined experiment, or the two bundled Figure 1 archives."""
    with open(PKL if results_path is None else results_path, "rb") as handle:
        results = pickle.load(handle)
    if results_path is not None:
        for dataset, *_ in MAIN_DATASETS:
            models = results[dataset]["models"]["gcn"]
            if not models or any(
                len(record["geometry"]) != len(LAYERS)
                or any("mad_global" not in layer for layer in record["geometry"])
                for record in models.values()
            ):
                raise ValueError(f"{dataset}: expected combined DE/global-MAD records")
        return results

    with open(GLOBAL_MAD_PKL, "rb") as handle:
        global_mad_results = pickle.load(handle)

    for dataset, *_ in MAIN_DATASETS:
        paired_models = results[dataset]["models"]["gcn"]
        global_models = global_mad_results[dataset]["models"]["gcn"]
        for seed, paired_record in paired_models.items():
            global_values = global_models[seed]["mad"]
            geometry = paired_record["geometry"]
            if len(global_values) != len(geometry):
                raise ValueError(
                    f"Layer mismatch for {dataset}, seed {seed}: "
                    f"{len(global_values)} global-MAD values versus "
                    f"{len(geometry)} geometry records"
                )
            for layer_record, global_mad in zip(geometry, global_values):
                layer_record["mad_global"] = global_mad

    return results


def build_figure(results_path=None) -> plt.Figure:
    configure()
    return render_main(load_results(results_path), MAIN_DATASETS)
