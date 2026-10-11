#!/usr/bin/env python3
"""Train models and measure probes and R_Group for the 24-setting comparison.

For each of the 12 multiclass datasets, two architectures, and five canonical
seeds, this script trains a four-layer model, fits independent linear probes,
and measures training-only R_Group at every layer.

Multi-split datasets pair canonical seed i with split i by default. Results
are saved atomically under output/controls/ after every model, and the run
can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import time

import torch

from src.linear_probe import linear_probing
from experiments.run_real import SEEDS, build_and_train, load_with_retry, set_seed
from src.geometry_metrics import group_distance_ratio


DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "amazon-photo",
    "amazon-computer",
    "coauthor-cs",
    "coauthor-physics",
    "wikics",
    "amazon-ratings",
    "chameleon",
    "squirrel",
    "roman-empire",
]
ARCHITECTURES = ["gcn", "gat"]
MULTI_SPLIT_DATASETS = {
    "wikics",
    "amazon-ratings",
    "chameleon",
    "squirrel",
    "roman-empire",
}


def atomic_pickle_dump(value, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json_dump(value, path: str) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pickle_or(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "rb") as handle:
        return pickle.load(handle)


def split_metadata(data: dict, run_idx: int) -> dict[str, int]:
    return {
        "run_idx": int(run_idx),
        "train_nodes": int(data["train_mask"].sum().item()),
        "valid_nodes": int(data["valid_mask"].sum().item()),
        "test_nodes": int(data["test_mask"].sum().item()),
    }


def record_is_complete(record: dict | None, depth: int) -> bool:
    if not record or len(record.get("probe", [])) != depth + 1:
        return False
    layers = record.get("controls", [])
    if len(layers) != depth + 1:
        return False
    return all("r_group" in layer for layer in layers)


@torch.no_grad()
def measure_controls(
    model,
    data: dict,
    input_control: dict | None = None,
) -> list[dict]:
    model.eval()
    _, hidden_states = model(data["X"], return_hidden=True)
    layers = []
    for layer, representation in enumerate(hidden_states):
        if layer == 0 and input_control is not None:
            layers.append({key: input_control[key] for key in ("layer", "r_group", "seconds")})
            print(
                f"    L0: R_Group={input_control['r_group']['ratio']:.6g} (cached)",
                flush=True,
            )
            continue
        started = time.time()
        r_group = group_distance_ratio(
            representation,
            data["labels"],
            data["train_mask"],
        )
        layer_record = {
            "layer": layer,
            "r_group": r_group,
            "seconds": time.time() - started,
        }
        layers.append(layer_record)
        print(
            f"    L{layer}: R_Group={r_group['ratio']:.6g}",
            flush=True,
        )
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=ARCHITECTURES,
        choices=ARCHITECTURES,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--paired-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pair seed i with split i for multi-split datasets (default: true)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="output/controls")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute requested records even if their saved output is complete",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="also save each trained GNN state_dict for geometric diagnostics",
    )
    args = parser.parse_args()

    unknown_seeds = sorted(set(args.seeds) - set(SEEDS))
    if unknown_seeds:
        parser.error(f"unknown seeds {unknown_seeds}; choose a subset of {SEEDS}")
    if args.depth < 1:
        parser.error("--depth must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    return args


def load_results(args, result_path: str) -> dict:
    """Load saved records and reject incompatible experiment configurations."""
    results = load_pickle_or(result_path, {"datasets": {}})
    results.setdefault("datasets", {})
    previous_configuration = results.get("configuration")
    if previous_configuration:
        incompatible = {
            "depth": args.depth,
            "paired_splits": bool(args.paired_splits),
        }
        conflicts = {
            key: (previous_configuration.get(key), value)
            for key, value in incompatible.items()
            if previous_configuration.get(key) != value
        }
        if conflicts:
            raise ValueError(
                f"existing output has incompatible configuration {conflicts}; "
                "use a new --out directory"
            )
    results["schema_version"] = 1
    results["configuration"] = {
        "depth": args.depth,
        "r_group_nodes": "training only",
        "r_group_distance": "squared Euclidean after row L2 normalization",
        "paired_splits": bool(args.paired_splits),
    }
    return results


def initial_metadata(args, device: str, configuration: dict) -> dict:
    """Describe the experiment configuration and execution environment."""
    return {
        "status": "running",
        "datasets": args.datasets,
        "architectures": args.architectures,
        "seeds": args.seeds,
        "configuration": configuration,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def run_models(args, device, results, metadata, result_path, metadata_path) -> int:
    """Run pending models in dataset/architecture/seed order, saving each one."""
    checkpoint_dir = os.path.join(args.out, "checkpoints")
    total = len(args.datasets) * len(args.architectures) * len(args.seeds)
    completed = 0
    input_control_cache: dict[tuple[str, int], dict] = {}
    for dataset in args.datasets:
        dataset_entry = results["datasets"].setdefault(dataset, {"models": {}})
        for architecture in args.architectures:
            architecture_entry = dataset_entry["models"].setdefault(architecture, {})
            for seed in args.seeds:
                existing = architecture_entry.get(seed)
                if not args.force and record_is_complete(existing, args.depth):
                    completed += 1
                    print(
                        f"skip {dataset} {architecture} seed={seed} "
                        f"({completed}/{total})",
                        flush=True,
                    )
                    continue

                run_idx = (
                    SEEDS.index(seed)
                    if args.paired_splits and dataset in MULTI_SPLIT_DATASETS
                    else 0
                )
                print(
                    f"=== {dataset} {architecture} seed={seed} split={run_idx} ===",
                    flush=True,
                )
                data = load_with_retry(dataset, run_idx, device)
                dataset_entry["num_nodes"] = int(data["num_nodes"])
                dataset_entry["num_classes"] = int(data["num_classes"])
                set_seed(seed)
                started = time.time()
                model = build_and_train(architecture, data, args.depth, device)
                probe = linear_probing(
                    model,
                    data["X"],
                    data["labels"],
                    data["train_mask"],
                    data["test_mask"],
                    data["num_classes"],
                )
                cache_key = (dataset, run_idx)
                controls = measure_controls(
                    model,
                    data,
                    input_control=input_control_cache.get(cache_key),
                )
                input_control_cache.setdefault(cache_key, controls[0])
                architecture_entry[seed] = {
                    "seed": int(seed),
                    "split": split_metadata(data, run_idx),
                    "probe": probe,
                    "controls": controls,
                    "training_seconds": time.time() - started,
                }
                if args.save_checkpoints:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{dataset}_{architecture}_depth{args.depth}_seed{seed}.pt",
                    )
                    torch.save(model.state_dict(), checkpoint_path)
                    architecture_entry[seed]["checkpoint"] = checkpoint_path

                atomic_pickle_dump(results, result_path)
                completed += 1
                metadata["completed_models"] = completed
                metadata["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                atomic_json_dump(metadata, metadata_path)
                print(
                    f"done {dataset} {architecture} seed={seed} "
                    f"({completed}/{total}) in {time.time() - started:.1f}s",
                    flush=True,
                )
                del model, data, probe, controls
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return completed


def main() -> None:
    args = parse_args()
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    os.makedirs(args.out, exist_ok=True)
    result_path = os.path.join(args.out, "supervised_control_results.pkl")
    metadata_path = os.path.join(args.out, "metadata.json")
    results = load_results(args, result_path)
    metadata = initial_metadata(args, device, results["configuration"])
    atomic_json_dump(metadata, metadata_path)

    total = len(args.datasets) * len(args.architectures) * len(args.seeds)
    print(f"device: {device}", flush=True)
    print(f"models requested: {total}", flush=True)
    print(f"paired splits: {args.paired_splits}", flush=True)
    print(f"output: {result_path}", flush=True)
    completed = run_models(args, device, results, metadata, result_path, metadata_path)

    metadata["status"] = "complete"
    metadata["completed_models"] = completed
    metadata["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json_dump(metadata, metadata_path)
    print("All supervision-matched controls complete.", flush=True)


if __name__ == "__main__":
    main()
