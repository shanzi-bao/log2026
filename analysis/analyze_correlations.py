#!/usr/bin/env python3
"""Calculate endpoint changes, paired cluster-bootstrap intervals, and log-ratio sensitivity.

Each setting contains five four-layer runs. Accuracy changes are percentage
points; entropy changes are differences; geometric changes are final/input - 1.
Bootstrap draws resample datasets, retaining their paired GCN/GAT settings.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "cora", "citeseer", "pubmed", "amazon-photo", "amazon-computer",
    "coauthor-cs", "coauthor-physics", "wikics", "amazon-ratings",
    "chameleon", "squirrel", "roman-empire",
)
ARCHITECTURES = ("gcn", "gat")
SEEDS = (42, 123, 456, 789, 1024)
GEOMETRY = ("r_group", "dirichlet", "mad")
VALUE_COLUMNS = (
    "probe_accuracy_input", "probe_accuracy_final",
    "probe_entropy_input", "probe_entropy_final",
    "r_group_input", "r_group_final", "dirichlet_input", "dirichlet_final",
    "mad_input", "mad_final", "delta_probe_accuracy", "delta_probe_entropy",
    "delta_r_group_relative", "delta_dirichlet_relative", "delta_mad_relative",
)
# The order fixes the random-number stream used for each diagnostic.
SPECS = (
    ("Probe entropy", "delta_probe_entropy_mean"),
    ("R_Group (relative)", "delta_r_group_relative_mean"),
    ("Scale-normalized Dirichlet energy", "delta_dirichlet_relative_mean"),
    ("Global MAD", "delta_mad_relative_mean"),
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value, context):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def collect_runs(run_dir):
    expected = {(d, a, s) for d in DATASETS for a in ARCHITECTURES for s in SEEDS}
    indexed = {}
    for path in sorted(run_dir.glob("*.json")):
        record = json.loads(path.read_text())
        key = (record["dataset"], record["architecture"], int(record["seed"]))
        if key not in expected or key in indexed:
            raise ValueError(f"Unexpected or duplicate run: {key}")
        if path.name != f"{key[0]}_{key[1]}_seed{key[2]}.json":
            raise ValueError(f"Filename and run metadata disagree: {path}")
        checks = record["verification"]
        if checks.get("success", True) is not True or any(
            checks.get(name) is not True
            for name in ("all_layers_rgroup_match", "strict_state_dict_loaded")
        ):
            raise ValueError(f"{key}: checkpoint or diagnostic verification failed")
        probe, geometry = record["probe"], record["geometry"]
        for entries in (probe, geometry):
            if [entry["layer"] for entry in entries] != list(range(5)):
                raise ValueError(f"{key}: expected aligned layers 0 through 4")
        for entry in probe:
            accuracy = finite(entry["accuracy"], f"{key} accuracy")
            entropy = finite(entry["entropy_normalized"], f"{key} entropy")
            if not 0 <= accuracy <= 100 or not -1e-12 <= entropy <= 1 + 1e-12:
                raise ValueError(f"{key}: accuracy or normalized entropy outside its range")
        for entry in geometry:
            for metric in GEOMETRY:
                if finite(entry[metric], f"{key} {metric}") < 0:
                    raise ValueError(f"{key}: negative geometric diagnostic")
        row = {
            "dataset": key[0], "architecture": key[1], "seed": key[2],
            "run_idx": int(record["split"]["run_idx"]),
            "probe_accuracy_input": float(probe[0]["accuracy"]),
            "probe_accuracy_final": float(probe[-1]["accuracy"]),
            "probe_entropy_input": float(probe[0]["entropy_normalized"]),
            "probe_entropy_final": float(probe[-1]["entropy_normalized"]),
        }
        for metric in GEOMETRY:
            start, end = float(geometry[0][metric]), float(geometry[-1][metric])
            if start <= 0:
                raise ValueError(f"{key}: {metric} has a non-positive input denominator")
            row[f"{metric}_input"], row[f"{metric}_final"] = start, end
        row["delta_probe_accuracy"] = row["probe_accuracy_final"] - row["probe_accuracy_input"]
        row["delta_probe_entropy"] = row["probe_entropy_final"] - row["probe_entropy_input"]
        for metric in GEOMETRY:
            row[f"delta_{metric}_relative"] = finite(
                row[f"{metric}_final"] / row[f"{metric}_input"] - 1, f"{key} relative change"
            )
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"Expected 120 complete runs; missing: {sorted(expected - set(indexed))}")
    for dataset in DATASETS:
        for seed in SEEDS:
            splits = {indexed[dataset, architecture, seed]["run_idx"] for architecture in ARCHITECTURES}
            if len(splits) != 1:
                raise ValueError(f"{dataset}/{seed}: GCN and GAT use different split indices")
    return [indexed[key] for key in sorted(indexed)]


def aggregate(runs, metadata_path):
    metadata_rows = read_csv(metadata_path)
    metadata = {row["dataset"]: row for row in metadata_rows}
    if len(metadata_rows) != len(DATASETS) or set(metadata) != set(DATASETS):
        raise ValueError("Metadata must contain each of the 12 datasets exactly once")
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["dataset"], run["architecture"]].append(run)
    expected = {(d, a) for d in DATASETS for a in ARCHITECTURES}
    if set(grouped) != expected:
        raise ValueError("Expected all 24 dataset-architecture settings")
    settings = []
    for (dataset, architecture), group in sorted(grouped.items()):
        if len(group) != 5 or {row["seed"] for row in group} != set(SEEDS):
            raise ValueError(f"{dataset}/{architecture}: expected five distinct model seeds")
        row = {
            "dataset": dataset, "architecture": architecture, "display": metadata[dataset]["display"],
            "adjusted_homophily": finite(metadata[dataset]["adjusted_homophily"], dataset),
            "runs": 5, "probe_endpoint_layer": 4, "geometry_endpoint_layer": 4,
        }
        for column in VALUE_COLUMNS:
            values = np.asarray([run[column] for run in group], dtype=float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        settings.append(row)
    return settings


def correlation(x, y, method):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float((pearsonr if method == "pearson" else spearmanr)(x, y).statistic)


def interval(values):
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite bootstrap correlations")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def bootstrap_correlations(settings, draws=10_000, seed=20260826):
    datasets = sorted({row["dataset"] for row in settings})
    groups = {
        dataset: np.asarray([i for i, row in enumerate(settings) if row["dataset"] == dataset])
        for dataset in datasets
    }
    for dataset, indices in groups.items():
        if len(indices) != 2 or {settings[i]["architecture"] for i in indices} != set(ARCHITECTURES):
            raise ValueError(f"{dataset}: expected paired GCN and GAT settings")
    rng = np.random.default_rng(seed)
    correlations, log_correlations = [], []
    y_column = "delta_probe_accuracy_mean"
    y = np.asarray([row[y_column] for row in settings], dtype=float)
    for name, x_column in SPECS:
        x = np.asarray([row[x_column] for row in settings], dtype=float)
        variants = [(name, x_column, x, correlations)]
        if x_column in ("delta_dirichlet_relative_mean", "delta_mad_relative_mean"):
            ratios = 1 + x
            if np.any(ratios <= 0):
                raise ValueError(f"{name}: log-ratio sensitivity requires positive setting ratios")
            # Average the five run ratios first, then take the logarithm.
            variants.append((name, x_column.replace("relative_mean", "log_ratio"),
                             np.log(ratios), log_correlations))
        samples = [{method: np.empty(draws) for method in ("pearson", "spearman")}
                   for _ in variants]
        for draw in range(draws):
            sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
            indices = np.concatenate([groups[dataset] for dataset in sampled_datasets])
            # Log-ratio sensitivity reuses the same dataset draws as its diagnostic.
            for (_, _, values, _), sampled in zip(variants, samples):
                for method in sampled:
                    sampled[method][draw] = correlation(values[indices], y[indices], method)
        for (label, column, values, destination), sampled in zip(variants, samples):
            pearson_ci, spearman_ci = interval(sampled["pearson"]), interval(sampled["spearman"])
            destination.append({
                "comparison": label, "x": column, "y": y_column,
                "pearson_r": correlation(values, y, "pearson"),
                "pearson_ci_low": pearson_ci[0], "pearson_ci_high": pearson_ci[1],
                "spearman_rho": correlation(values, y, "spearman"),
                "spearman_ci_low": spearman_ci[0], "spearman_ci_high": spearman_ci[1],
                "dataset_clusters": len(datasets), "settings": len(settings),
                "bootstrap_draws": draws, "bootstrap_seed": seed,
            })
    return correlations, log_correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/correlations/runs")
    parser.add_argument("--metadata", type=Path, default=ROOT / "results/correlations/metadata.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/statistics/correlations")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == ROOT / "results" or (ROOT / "results") in args.output.parents:
        parser.error("--output must not overwrite the included results/")
    if args.draws <= 0:
        parser.error("--draws must be positive")
    runs = collect_runs(args.run_dir)
    settings = aggregate(runs, args.metadata)
    correlations, log_correlations = bootstrap_correlations(settings, args.draws, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("run_changes.csv", runs), ("setting_changes.csv", settings),
                           ("correlations.csv", correlations), ("log_ratio_correlations.csv", log_correlations)):
        write_csv(args.output / filename, rows)
    for row in correlations:
        print(f"{row['comparison']}: r={row['pearson_r']:.6f}, rho={row['spearman_rho']:.6f}")
    print(f"Saved 120 run changes, 24 setting means, and bootstrap results to {args.output}")


if __name__ == "__main__":
    main()
