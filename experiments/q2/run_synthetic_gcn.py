"""Run the controlled GraphUniverse GCN depth sweep.

Trains independent GCNs at total depths 2/4/6/8/10/12/16/20 on the matched
high- and low-homophily graphs. Stores layer-wise linear-probe trajectories
for five canonical model seeds.

Example:
    python -m experiments.q2.run_synthetic_gcn
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from src.gcn import SimpleGNN, normalize_adj, train_eval_loop_gnn_cora
from src.linear_probe import linear_probing
from src.synthetic_dataset import load_synthetic_graph


GRAPH_FILES = {
    "high": Path(__file__).resolve().parents[2] / "data/graphuniverse/highhomodata.pt",
    "mid": Path(__file__).resolve().parents[2] / "data/graphuniverse/midhomodata.pt",
    "low": Path(__file__).resolve().parents[2] / "data/graphuniverse/lowhomodata.pt",
}
CANONICAL_SEEDS = [42, 123, 456, 789, 1024]


def atomic_pickle_dump(value, path: str) -> None:
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()


def edge_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    source, target = edge_index
    return float((labels[source] == labels[target]).float().mean().item())


def train_gcn(data: dict, depth: int, device: str) -> SimpleGNN:
    model = SimpleGNN(
        data["X"].shape[1],
        32,
        data["num_classes"],
        depth,
        data["adj_norm"],
    ).to(device)
    train_eval_loop_gnn_cora(
        model,
        data["X"],
        data["train_y"],
        data["train_mask"],
        data["X"],
        data["valid_y"],
        data["valid_mask"],
        data["X"],
        data["test_y"],
        data["test_mask"],
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=sorted(GRAPH_FILES),
        default=["high", "low"],
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8, 10, 12, 16, 20],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=CANONICAL_SEEDS,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out",
        default="output/synthetic_gcn",
    )
    args = parser.parse_args()

    if min(args.depths) < 1 or len(set(args.depths)) != len(args.depths):
        parser.error("--depths must contain unique positive integers")
    return args


def make_metadata(args, graph_name, data, measured_homophily):
    return {
        "dataset": f"synthetic-{graph_name}",
        "model": "gcn",
        "protocol": "graphuniverse-native-60/20/20",
        "graph_idx": 0,
        "split_seed": 42,
        "edge_homophily": measured_homophily,
        "num_nodes": int(data["num_nodes"]),
        "num_classes": int(data["num_classes"]),
        "train_size": int(data["train_mask"].sum()),
        "requested_depths": sorted(args.depths),
        "requested_seeds": sorted(args.seeds),
        "training": {
            "epochs": 100,
            "learning_rate": 0.01,
            "hidden_dim": 32,
            "activation": "relu",
        },
        "environment": {
            "torch": torch.__version__,
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "status": "running",
    }


def run_graph(args, graph_name) -> None:
    data = load_synthetic_graph(
        GRAPH_FILES[graph_name],
        graph_idx=0,
        seed=42,
        device=args.device,
    )
    edge_index = data["A"].nonzero().t().contiguous()
    data["edge_index"] = edge_index
    data["adj_norm"] = normalize_adj(
        edge_index=edge_index,
        num_nodes=data["num_nodes"],
    )
    measured_homophily = edge_homophily(edge_index, data["labels"])

    out_dir = os.path.join(
        args.out, f"gcn_h32_synthetic_{graph_name}"
    )
    os.makedirs(out_dir, exist_ok=True)
    probe_path = os.path.join(out_dir, "probe_results.pkl")
    meta_path = os.path.join(out_dir, "meta.json")

    results = load_pickle_or(probe_path, {})
    for seed in args.seeds:
        results.setdefault(seed, {})

    metadata = make_metadata(args, graph_name, data, measured_homophily)
    atomic_json_dump(metadata, meta_path)

    print(
        f"{graph_name}: nodes={data['num_nodes']} "
        f"classes={data['num_classes']} edge_h={measured_homophily:.3f}",
        flush=True,
    )

    for seed in args.seeds:
        for depth in args.depths:
            if depth in results[seed]:
                print(
                    f"skip graph={graph_name} seed={seed} depth={depth}",
                    flush=True,
                )
                continue

            set_seed(seed)
            started = time.time()
            model = train_gcn(data, depth, args.device)
            results[seed][depth] = linear_probing(
                model,
                data["X"],
                data["labels"],
                data["train_mask"],
                data["test_mask"],
                data["num_classes"],
            )
            atomic_pickle_dump(results, probe_path)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"done graph={graph_name} seed={seed} depth={depth} "
                f"seconds={time.time() - started:.1f}",
                flush=True,
            )

    metadata["completed"] = {
        str(seed): sorted(map(int, results.get(seed, {}).keys()))
        for seed in args.seeds
    }
    metadata["status"] = "complete"
    metadata["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json_dump(metadata, meta_path)
    del data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    print(
        f"device={args.device}; graphs={args.graphs}; "
        f"depths={args.depths}; seeds={args.seeds}",
        flush=True,
    )

    for graph_name in args.graphs:
        run_graph(args, graph_name)

    print("GCN SYNTHETIC DEPTH SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
