"""
Probe experiments on the 12 multiclass datasets from the
tunedGNN / Transductive-Sharpening benchmark, using the split protocol of
https://github.com/transductive-sharpening/tunedGNN.

Protocol per run i (i = 0..4, seed = SEEDS[i]):
  - cora/citeseer/pubmed: one fixed random class-balanced split (20 per class),
    model seed varies
  - amazon-*/coauthor-*: fixed 60/20/20 repo split, model seed varies
  - wikics / hetero / squirrel / chameleon(filtered): split #i, seed SEEDS[i]

Results are saved under output/real/ by default, with pickle structure
{seed: {num_layers: [per-layer dicts]}}. The default depths are 4 and 10.
Per-node entropy-confidence arrays are collected only at these two depths.

Usage:
    python -m experiments.run_real
    python -m experiments.run_real --datasets cora wikics --models gcn
"""

import argparse
import json
import os
import pickle
import platform
import time

import numpy as np
import torch

from src.gcn import SimpleGNN, train_eval_loop_gnn_cora
from src.gat import GAT, train_eval_loop_gat
from src.linear_probe import linear_probing, linear_probing_per_node
from src.my_datasets import TS_DATASETS, load_ts_dataset

SEEDS = [42, 123, 456, 789, 1024]
DATASETS = TS_DATASETS
PER_NODE_DEPTHS = (4, 10)


def atomic_pickle_dump(obj, path):
    """Write a pickle atomically so an interrupted job keeps the last checkpoint."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    """Write JSON atomically so metadata never marks a torn checkpoint."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_pickle_or(path, default):
    if not os.path.exists(path):
        return default
    with open(path, 'rb') as f:
        return pickle.load(f)


def results_dir_name(model_name, dataset_name):
    base = {'gcn': 'gcn_h32', 'gat': 'sgat4h_relu'}[model_name]
    return f"{base}_{dataset_name.lower().replace('-', '_')}"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()


def build_and_train(model_name, data, num_layers, device):
    if model_name == 'gcn':
        model = SimpleGNN(data['X'].shape[1], 32, data['num_classes'],
                          num_layers, data['adj_norm']).to(device)
        train_eval_loop_gnn_cora(model, data['X'], data['train_y'], data['train_mask'],
                                 data['X'], data['valid_y'], data['valid_mask'],
                                 data['X'], data['test_y'], data['test_mask'])
    elif model_name == 'gat':
        model = GAT(data['X'].shape[1], 8, data['num_classes'], num_layers,
                    data['edge_index'], num_heads=4).to(device)
        train_eval_loop_gat(model, data['X'], data['train_y'], data['train_mask'],
                            data['X'], data['valid_y'], data['valid_mask'],
                            data['X'], data['test_y'], data['test_mask'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def load_with_retry(name, run_idx, device, attempts=5, wait=30):
    """Load a dataset with retries for transient download failures."""
    for a in range(attempts):
        try:
            return load_ts_dataset(name, run_idx=run_idx, device=device)
        except Exception as e:
            if a == attempts - 1:
                raise
            print(f"  load {name} failed ({type(e).__name__}); "
                  f"retry {a + 1}/{attempts - 1} in {wait}s")
            time.sleep(wait)


def run_dataset(dataset_name, models, depths, seeds, device, out_root, skip_per_node=False):
    print(f"\n{'#'*70}\n# {dataset_name}\n{'#'*70}")
    per_node_depths = [] if skip_per_node else [
        depth for depth in depths if depth in PER_NODE_DEPTHS
    ]

    for model_name in models:
        out_dir = os.path.join(out_root, results_dir_name(model_name, dataset_name))
        os.makedirs(out_dir, exist_ok=True)

        probe_path = os.path.join(out_dir, 'probe_results.pkl')
        per_node_path = os.path.join(out_dir, 'auc_results.pkl')
        meta_path = os.path.join(out_dir, 'meta.json')
        probe_results = load_pickle_or(probe_path, {})
        per_node_results = load_pickle_or(per_node_path, {}) if per_node_depths else {}
        for seed in seeds:
            probe_results.setdefault(seed, {})
            if per_node_depths:
                per_node_results.setdefault(seed, {})

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {'dataset': dataset_name, 'model': model_name,
                    'protocol': 'tunedGNN-aligned', 'runs': []}
        meta['requested_depths'] = sorted(set(meta.get('requested_depths', [])) | set(depths))
        meta['requested_seeds'] = sorted(set(meta.get('requested_seeds', [])) | set(seeds))
        meta.pop('skip_auc', None)
        meta.pop('completed_auc', None)
        meta['skip_per_node'] = bool(skip_per_node)
        meta['requested_per_node_depths'] = sorted(per_node_depths)
        meta['training'] = {
            'epochs': 100, 'learning_rate': 0.01, 'hidden_dim': 32,
            'activation': 'relu',
            'gat_heads': 4 if model_name == 'gat' else None,
        }
        meta['environment'] = {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        run_meta = {int(r['seed']): r for r in meta.get('runs', [])}

        for seed in seeds:
            run_idx = SEEDS.index(seed)
            pending = [nl for nl in depths
                       if nl not in probe_results[seed]
                       or (nl in per_node_depths and nl not in per_node_results[seed])]
            if not pending:
                print(f"  {model_name.upper()} run={run_idx} seed={seed} "
                      f"depths already complete — skipping")
                continue

            data = load_with_retry(dataset_name, run_idx, device)
            run_meta[seed] = {'seed': seed, 'run_idx': run_idx,
                              'num_nodes': data['num_nodes'],
                              'num_classes': data['num_classes'],
                              'train_size': int(data['train_mask'].sum()),
                              'valid_size': int(data['valid_mask'].sum()),
                              'test_size': int(data['test_mask'].sum())}
            meta['runs'] = sorted(run_meta.values(), key=lambda r: r['run_idx'])

            for nl in pending:
                set_seed(seed)
                model = build_and_train(model_name, data, nl, device)
                needs_per_node = (
                    nl in per_node_depths and nl not in per_node_results[seed]
                )
                # Fit aggregate probes first so the per-node pass uses the prescribed RNG state.
                if nl not in probe_results[seed] or needs_per_node:
                    aggregate = linear_probing(
                        model, data['X'], data['labels'],
                        data['train_mask'], data['test_mask'], data['num_classes'])
                    if nl not in probe_results[seed]:
                        probe_results[seed][nl] = aggregate
                if needs_per_node:
                    per_node_results[seed][nl] = linear_probing_per_node(
                        model, data['X'], data['labels'],
                        data['train_mask'], data['test_mask'], data['num_classes'])
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  {model_name.upper()} {nl}L run={run_idx} seed={seed} "
                      f"done ({dataset_name})")

                atomic_pickle_dump(probe_results, probe_path)
                if per_node_depths:
                    atomic_pickle_dump(per_node_results, per_node_path)
                meta['completed_probe'] = {
                    str(s): sorted(map(int, probe_results.get(s, {}).keys()))
                    for s in seeds
                }
                if per_node_depths:
                    meta['completed_per_node'] = {
                        str(s): sorted(map(int, per_node_results.get(s, {}).keys()))
                        for s in seeds
                    }
                meta['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                meta['status'] = 'running'
                atomic_json_dump(meta, meta_path)
            del data

        complete_probe = all(nl in probe_results.get(seed, {})
                             for seed in seeds for nl in depths)
        complete_per_node = all(nl in per_node_results.get(seed, {})
                                for seed in seeds for nl in per_node_depths)
        meta['status'] = 'complete' if complete_probe and complete_per_node else 'partial'
        meta['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        atomic_json_dump(meta, meta_path)
        print(f"Saved {model_name.upper()} results to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description='tunedGNN-aligned probe experiments')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                        choices=DATASETS)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['gcn', 'gat'], choices=['gcn', 'gat'])
    parser.add_argument('--depths', type=int, nargs='+', default=[4, 10])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='subset of canonical seeds: 42 123 456 789 1024')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out', type=str, default='output/real')
    parser.add_argument('--skip-per-node', action='store_true',
                        help='skip the second probe pass for per-node alignment at depths 4 and 10')
    args = parser.parse_args()

    unknown_seeds = sorted(set(args.seeds) - set(SEEDS))
    if unknown_seeds:
        parser.error(f"unknown seeds {unknown_seeds}; choose a subset of {SEEDS}")
    if len(set(args.depths)) != len(args.depths) or min(args.depths) < 1:
        parser.error('--depths must contain unique positive integers')

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    print(f"Using device: {device}")
    print(f"Datasets ({len(args.datasets)}): {args.datasets}")
    print(f"Depths: {args.depths}; seeds: {args.seeds}; skip_per_node={args.skip_per_node}")

    for dataset_name in args.datasets:
        run_dataset(dataset_name, args.models, args.depths, args.seeds,
                    device, args.out, skip_per_node=args.skip_per_node)

    print("\nAll tunedGNN-aligned probe experiments complete.")


if __name__ == '__main__':
    main()
