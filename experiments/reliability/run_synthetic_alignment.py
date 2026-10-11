"""Two-pass probe alignment on the three GraphUniverse synthetic graphs.

Native 60/20/20 split (graph_idx=0, split seed 42 fixed; model seeds vary),
GCN + GAT, depths 4 and 10, 5 model seeds, CPU.
Prints measured edge homophily as the protocol check (expect ~0.865/0.512/0.110).

The aggregate probe pass is followed by an independently fitted per-node pass.
Their order and random-number stream are fixed for each model.

Usage: python -m experiments.reliability.run_synthetic_alignment
Writes output/synthetic_alignment/<prefix>_synthetic_<name>/.
"""
import argparse
import json
import os
from pathlib import Path
import pickle

import torch

from src.gcn import normalize_adj
from src.linear_probe import linear_probing, linear_probing_per_node
from experiments.run_real import SEEDS, build_and_train, set_seed
from src.synthetic_dataset import load_synthetic_graph

DATA_DIR = Path(__file__).resolve().parents[2] / 'data/graphuniverse'
GRAPHS = {'high': DATA_DIR / 'highhomodata.pt',
          'mid': DATA_DIR / 'midhomodata.pt',
          'low': DATA_DIR / 'lowhomodata.pt'}
DEPTHS = [4, 10]
MODELS = [('gcn', 'gcn_h32'), ('gat', 'sgat4h_relu')]


def edge_homophily(edge_index, labels):
    src, dst = edge_index
    return (labels[src] == labels[dst]).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='output/synthetic_alignment')
    args = parser.parse_args()
    device = 'cpu'
    for name, path in GRAPHS.items():
        data = load_synthetic_graph(path, graph_idx=0, seed=42, device=device)
        edge_index = data['A'].nonzero().t().contiguous()
        data['edge_index'] = edge_index
        data['adj_norm'] = normalize_adj(edge_index=edge_index,
                                         num_nodes=data['num_nodes'])
        h = edge_homophily(edge_index, data['labels'])
        print(f"synthetic-{name}: nodes={data['num_nodes']} "
              f"classes={data['num_classes']} edge_h={h:.3f} "
              f"train={int(data['train_mask'].sum())}")
        for model_name, prefix in MODELS:
            out_dir = os.path.join(args.out,
                                   f'{prefix}_synthetic_{name}')
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(os.path.join(out_dir, 'meta.json')):
                print(f"  {model_name} already complete — skip")
                continue
            probe_results = {s: {} for s in SEEDS}
            per_node_results = {s: {} for s in SEEDS}
            meta = {'dataset': f'synthetic-{name}', 'model': model_name,
                    'protocol': 'graphuniverse-native-60/20/20',
                    'graph_idx': 0, 'split_seed': 42, 'edge_homophily': h,
                    'num_nodes': data['num_nodes'],
                    'num_classes': data['num_classes'],
                    'train_size': int(data['train_mask'].sum())}
            for seed in SEEDS:
                for nl in DEPTHS:
                    set_seed(seed)
                    model = build_and_train(model_name, data, nl, device)
                    # Fit aggregate probes first so the per-node pass uses the prescribed RNG state.
                    probe_results[seed][nl] = linear_probing(
                        model, data['X'], data['labels'],
                        data['train_mask'], data['test_mask'], data['num_classes'])
                    per_node_results[seed][nl] = linear_probing_per_node(
                        model, data['X'], data['labels'],
                        data['train_mask'], data['test_mask'], data['num_classes'])
                    del model
                    print(f"  {model_name} {nl}L seed={seed} done")
                    with open(os.path.join(out_dir, 'probe_results.pkl'), 'wb') as f:
                        pickle.dump(probe_results, f)
                    with open(os.path.join(out_dir, 'auc_results.pkl'), 'wb') as f:
                        pickle.dump(per_node_results, f)
            with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
    print("SYNTHETIC PROBE DONE")


if __name__ == '__main__':
    main()
