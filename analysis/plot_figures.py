"""Generate all six paper figures from the included experimental results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plots import (
    correlations,
    depth_summary,
    failure_onset,
    real_trajectories,
    smoothing,
    synthetic_trajectories,
)
from .plots.style import normalize

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    ("main/fig1_smoothing_consequences", 2.85, smoothing.build_figure),
    ("main/fig2_failure_onset", 2.40, failure_onset.build_figure),
    ("appendix/figA1_all_settings_validation", 5.1 * 5.5 / 6.3,
     correlations.build_figure),
    ("appendix/figA3_failure_onset_summary", 136.455 * 5.5 / 411.395,
     depth_summary.build_figure),
    ("appendix/figA4_gat_failure_onset_trajectories", 237.383 * 5.5 / 514.620,
     synthetic_trajectories.build_figure),
    ("appendix/figA5b_real_layerwise_trajectories_cora", 4.15,
     real_trajectories.build_figure),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "figures",
                        help="Directory for generated PDFs and PNGs.")
    parser.add_argument("--figure1-results", type=Path,
                        help="Combined Figure 1 pickle from experiments.q1.run_smoothing_trajectories.")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "results" or (ROOT / "results") in output.parents:
        parser.error("--output must not be inside results/")
    for number, (stem, height, build) in enumerate(FIGURES, 1):
        figure = (
            smoothing.build_figure(args.figure1_results)
            if number == 1 and args.figure1_results is not None else build()
        )
        figure.canvas.draw()
        normalize(figure, number, height)
        target = output / stem
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target.with_suffix(".pdf"), facecolor="white")
        figure.savefig(target.with_suffix(".png"), dpi=200, facecolor="white")
        plt.close(figure)
        print(target.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
