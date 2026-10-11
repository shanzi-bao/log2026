#!/usr/bin/env python3
"""Summarize real and synthetic per-node entropy-confidence alignment.

The real-data table averages five runs per dataset, then reports the mean and
sample standard deviation across datasets (ddof=1). The synthetic table reports
the mean and population standard deviation across five runs (ddof=0).
Pickle inputs must come from trusted sources.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "coauthor_physics", "amazon_photo", "coauthor_cs", "cora", "pubmed",
    "amazon_computer", "citeseer", "wikics", "amazon_ratings",
    "chameleon", "squirrel", "roman_empire",
)
MODELS = {"GCN": "gcn_h32_", "GAT": "sgat4h_relu_"}
SEEDS = (42, 123, 456, 789, 1024)
DEPTHS = (4, 10)
REGIMES = {"high": "High", "mid": "Intermediate", "low": "Low"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_node_spearman(layer_records, depth, expected_nodes=None):
    if [row["layer"] for row in layer_records] != list(range(depth + 1)):
        raise ValueError(f"Expected consecutive layers 0 through {depth}")
    entropy = np.stack([row["entropy_per_node"] for row in layer_records])
    nll = np.stack([row["neg_log_p_true_per_node"] for row in layer_records])
    if entropy.ndim != 2 or entropy.shape != nll.shape or not entropy.shape[1]:
        raise ValueError("Expected matching layer-by-node entropy and log-probability arrays")
    if expected_nodes is not None and entropy.shape[1] != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} evaluation nodes, found {entropy.shape[1]}")
    if not np.isfinite(entropy).all() or not np.isfinite(nll).all():
        raise ValueError("Per-node arrays contain non-finite values")
    entropy_rank = rankdata(entropy, axis=0).astype(np.float64)
    nll_rank = rankdata(nll, axis=0).astype(np.float64)
    entropy_rank -= entropy_rank.mean(0)
    nll_rank -= nll_rank.mean(0)
    numerator = (entropy_rank * nll_rank).sum(0)
    denominator = np.sqrt((entropy_rank ** 2).sum(0) * (nll_rank ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = numerator / denominator
    # A constant layer trajectory has undefined Spearman correlation.
    valid = rho[np.isfinite(rho)]
    if not valid.size:
        raise ValueError("No nodes have a defined Spearman correlation")
    return float(valid.mean())


def load_archive(path):
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if set(records) != set(SEEDS):
        raise ValueError(f"{path}: expected the five model seeds {SEEDS}")
    for seed in SEEDS:
        if not set(DEPTHS).issubset(records[seed]):
            raise ValueError(f"{path}: missing four-layer or ten-layer results for seed {seed}")
    return records


def real_runs_from_raw(root):
    rows = []
    for dataset in DATASETS:
        for architecture, prefix in MODELS.items():
            records = load_archive(root / (prefix + dataset) / "auc_results.pkl")
            for depth in DEPTHS:
                for seed in SEEDS:
                    rows.append({
                        "dataset": dataset, "architecture": architecture, "depth": depth,
                        "seed": seed, "rho_bar": per_node_spearman(records[seed][depth], depth),
                    })
    return rows


def summarize_real(rows):
    indexed = {(row["dataset"], row["architecture"], int(row["depth"]), int(row["seed"])):
               float(row["rho_bar"]) for row in rows}
    expected = {(d, a, k, s) for d in DATASETS for a in MODELS for k in DEPTHS for s in SEEDS}
    if len(rows) != 240 or set(indexed) != expected or not all(np.isfinite(v) for v in indexed.values()):
        raise ValueError("Expected 240 unique, finite real-data alignment values")
    summary = []
    for regime, datasets in (("Homophilous", DATASETS[:8]), ("Heterophilous", DATASETS[8:])):
        for architecture in MODELS:
            for depth in DEPTHS:
                values = np.asarray([
                    np.mean([indexed[dataset, architecture, depth, seed] for seed in SEEDS])
                    for dataset in datasets
                ])
                summary.append({
                    "regime": regime, "datasets": len(datasets), "architecture": architecture,
                    "depth": depth, "mean": float(values.mean()),
                    "sample_sd_across_datasets": float(values.std(ddof=1)),
                })
    return summary


def synthetic_runs(root):
    rows = []
    for regime, label in REGIMES.items():
        for architecture, prefix in MODELS.items():
            directory = root / f"{prefix}synthetic_{regime}"
            metadata = json.loads((directory / "meta.json").read_text())
            if metadata["dataset"] != f"synthetic-{regime}" or metadata["model"] != architecture.lower():
                raise ValueError(f"{directory}: dataset or architecture metadata disagree")
            if int(metadata["num_nodes"]) != 500 or int(metadata["num_classes"]) != 10:
                raise ValueError(f"{directory}: expected 500 nodes and 10 classes")
            records = load_archive(directory / "auc_results.pkl")
            for depth in DEPTHS:
                for seed in SEEDS:
                    rows.append({
                        "regime": label, "edge_homophily": float(metadata["edge_homophily"]),
                        "architecture": architecture, "depth": depth, "seed": seed,
                        "rho_bar": per_node_spearman(records[seed][depth], depth, expected_nodes=100),
                    })
    return rows


def summarize_synthetic(rows):
    indexed = {(row["regime"], row["architecture"], int(row["depth"]), int(row["seed"])): row
               for row in rows}
    expected = {(r, a, k, s) for r in REGIMES.values() for a in MODELS for k in DEPTHS for s in SEEDS}
    if len(rows) != 60 or set(indexed) != expected:
        raise ValueError("Expected 60 unique synthetic alignment values")
    summary = []
    for regime in REGIMES.values():
        homophilies = {float(row["edge_homophily"]) for row in rows if row["regime"] == regime}
        if len(homophilies) != 1:
            raise ValueError(f"{regime}: inconsistent graph homophily across runs")
        homophily = homophilies.pop()
        for architecture in MODELS:
            for depth in DEPTHS:
                values = np.asarray([float(indexed[regime, architecture, depth, seed]["rho_bar"])
                                     for seed in SEEDS])
                if not np.isfinite(values).all():
                    raise ValueError(f"{regime}/{architecture}/{depth}: non-finite alignment")
                summary.append({
                    "regime": regime, "edge_homophily": homophily, "architecture": architecture,
                    "depth": depth, "runs": len(SEEDS), "mean": float(values.mean()),
                    "population_sd_across_runs": float(values.std(ddof=0)),
                })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-input", type=Path, default=ROOT / "results/probe_alignment/per_run.csv")
    parser.add_argument("--real-raw-dir", type=Path, help="Recalculate real-data per-run values from trusted pickle archives")
    parser.add_argument("--synthetic-dir", type=Path, default=ROOT / "results/synthetic_alignment")
    parser.add_argument("--output", type=Path, default=ROOT / "output/statistics")
    parser.add_argument("--only", choices=("real", "synthetic", "both"), default="both")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == ROOT / "results" or (ROOT / "results") in args.output.parents:
        parser.error("--output must not overwrite the included results/")
    if args.only in ("real", "both"):
        rows = real_runs_from_raw(args.real_raw_dir) if args.real_raw_dir else read_csv(args.real_input)
        summary = summarize_real(rows)
        write_csv(args.output / "probe_alignment/per_run.csv", rows)
        write_csv(args.output / "probe_alignment/summary.csv", summary)
        for row in summary:
            print(f"{row['regime']} {row['architecture']} {row['depth']} layers: "
                  f"{row['mean']:.2f} +/- {row['sample_sd_across_datasets']:.2f}")
    if args.only in ("synthetic", "both"):
        rows = synthetic_runs(args.synthetic_dir)
        summary = summarize_synthetic(rows)
        write_csv(args.output / "synthetic_alignment/per_run.csv", rows)
        write_csv(args.output / "synthetic_alignment/summary.csv", summary)
        for row in summary:
            print(f"{row['regime']} {row['architecture']} {row['depth']} layers: "
                  f"{row['mean']:.2f} +/- {row['population_sd_across_runs']:.2f}")


if __name__ == "__main__":
    main()
