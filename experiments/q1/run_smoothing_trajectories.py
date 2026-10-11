"""Measure Figure 1 probe, Dirichlet-energy, and global-MAD trajectories.

The default experiment uses four-layer GCNs on Cora, Amazon-Ratings, and
Roman-Empire. Each GNN is trained once before its independent linear probes.
Both geometric diagnostics use the same frozen layer representations.

Usage: python -m experiments.q1.run_smoothing_trajectories --paired-splits
Results are saved to output/geometry/geometry_grid_results.pkl.
For benchmarks with multiple splits, --paired-splits pairs model seed i
with split i. Pickle inputs must come from trusted sources.
"""
import argparse
import os
import pickle

import torch

from src.linear_probe import linear_probing
from src.geometry_metrics import global_mean_average_distance
from src.my_datasets import load_ts_dataset
from experiments.run_real import SEEDS, build_and_train, set_seed

DATASETS = ['cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer',
            'coauthor-cs', 'coauthor-physics', 'wikics',
            'amazon-ratings', 'chameleon', 'squirrel', 'roman-empire']
MULTI_SPLIT_DATASETS = {
    'wikics', 'amazon-ratings', 'chameleon', 'squirrel', 'roman-empire'
}
ARCHS = ['gcn', 'gat']
PAPER_DATASETS = ['cora', 'amazon-ratings', 'roman-empire']
DEPTH = 4
OUT = 'output/geometry'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=PAPER_DATASETS,
                        choices=DATASETS)
    parser.add_argument('--architectures', nargs='+', default=['gcn'],
                        choices=ARCHS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--out', default=OUT)
    parser.add_argument('--base-pkl', default=None,
                        help='Optional complete result pickle used to seed a '
                             'new output directory.')
    parser.add_argument('--paired-splits', action='store_true',
                        help='Pair canonical seed i with dataset split i.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute selected datasets even if present.')
    parser.add_argument('--device', default='auto',
                        choices=('auto', 'cpu', 'cuda'))
    args = parser.parse_args()
    if args.paired_splits:
        unknown = [seed for seed in args.seeds if seed not in SEEDS]
        if unknown:
            parser.error('--paired-splits requires canonical seeds from '
                         f'{list(SEEDS)}; got {unknown}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda requested, but CUDA is unavailable')
    return args


def assert_split_only_change(base, run, dataset, run_idx):
    """Ensure changing run_idx changes only masks, not graph contents."""
    for key in ('num_nodes', 'num_classes'):
        if base[key] != run[key]:
            raise ValueError(
                f'{dataset} split {run_idx} changed {key}: '
                f'{base[key]} != {run[key]}')
    for key in ('X', 'labels', 'edge_index'):
        if not torch.equal(base[key], run[key]):
            raise ValueError(
                f'{dataset} split {run_idx} changed {key}; only split masks '
                'may vary across run_idx')


def split_metadata(data, run_idx):
    return {
        'run_idx': int(run_idx),
        'train_nodes': int(data['train_mask'].sum().item()),
        'valid_nodes': int(data['valid_mask'].sum().item()),
        'test_nodes': int(data['test_mask'].sum().item()),
    }


def save_results(results, path):
    """Atomically replace the checkpoint so interruption cannot corrupt it."""
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(results, f)
    os.replace(tmp, path)


EDGE_CHUNK = 200_000


def _chunks(m):
    for a in range(0, m, EDGE_CHUNK):
        yield a, min(a + EDGE_CHUNK, m)


def chunked_dirichlet(h, edge_index):
    """Same math as geometry_metrics.dirichlet_energy (deg-normalized, divided
    by mean squared node norm), edge-chunked for large feature matrices."""
    hf = h.detach().float()
    src, dst = edge_index
    deg = torch.zeros(hf.size(0), device=hf.device).scatter_add_(
        0, dst, torch.ones(dst.numel(), device=hf.device))
    hs = hf / (1.0 + deg).sqrt().unsqueeze(1)
    tot = 0.0
    for a, b in _chunks(src.numel()):
        tot += float((hs[src[a:b]] - hs[dst[a:b]]).pow(2).sum().item())
    energy = tot / src.numel()
    scale = float(hf.pow(2).sum(dim=1).mean().clamp_min(1e-12).item())
    return energy / scale


@torch.no_grad()
def measure_geometry(hidden_states, edge_index, input_dirichlet=None):
    """Measure DE and global MAD without modifying representations or RNG state."""
    return [
        {
            'layer': k,
            'dirichlet': (
                input_dirichlet if k == 0 and input_dirichlet is not None
                else chunked_dirichlet(h, edge_index)
            ),
            'mad_global': global_mean_average_distance(h),
        }
        for k, h in enumerate(hidden_states)
    ]


def entry_is_complete(entry, architectures, seeds):
    """Require both geometric diagnostics and probe readouts at all five layers."""
    if not architectures or not seeds:
        return False
    for architecture in architectures:
        runs = entry.get('models', {}).get(architecture, {})
        for seed in seeds:
            record = runs.get(seed, {})
            for field, required in (
                ('probe', {'accuracy', 'entropy_normalized'}),
                ('geometry', {'dirichlet', 'mad_global'}),
            ):
                layers = record.get(field, [])
                if len(layers) != DEPTH + 1 or any(
                    row.get('layer') != k or not required.issubset(row)
                    for k, row in enumerate(layers)
                ):
                    return False
    return True


def load_results(args, out_pkl):
    """Load an existing archive and reject incomplete geometric records."""
    results = {}
    if os.path.exists(out_pkl):
        with open(out_pkl, 'rb') as f:
            results = pickle.load(f)
        print(f"resuming {len(results)} datasets from {out_pkl}")
    elif args.base_pkl:
        with open(args.base_pkl, 'rb') as f:
            results = pickle.load(f)
        print(f"loaded {len(results)} base datasets from {args.base_pkl}")
    for ds, entry in results.items():
        if args.force and ds in args.datasets:
            continue
        models = entry.get('models', {})
        if not models or any(
            not entry_is_complete(entry, [arch], list(runs))
            for arch, runs in models.items()
        ):
            raise ValueError(
                f'{ds}: archive lacks complete probe/DE/global-MAD records; '
                'use a new --out directory without --base-pkl, or --force '
                'with all incomplete datasets selected.')
    return results


def should_skip_dataset(args, ds, results):
    """Check resume compatibility and prevent partial replacement of a dataset."""
    if ds not in results:
        return False
    if not args.force:
        expected_paired = bool(args.paired_splits and ds in MULTI_SPLIT_DATASETS)
        if results[ds].get('paired_splits') != expected_paired:
            raise ValueError(
                f'{ds}: stored split protocol differs; use a new --out directory.')
        if not entry_is_complete(results[ds], args.architectures, args.seeds):
            raise ValueError(
                f'{ds}: saved results lack complete probe/DE/global-MAD records; '
                'use a new --out directory or --force to recompute the dataset.')
        return True
    stored_models = results[ds]['models']
    missing_architectures = set(stored_models) - set(args.architectures)
    missing_seeds = {
        seed for runs in stored_models.values() for seed in runs
    } - set(args.seeds)
    if missing_architectures or missing_seeds:
        raise ValueError(
            'Replacing a dataset requires all architectures and seeds '
            'already stored for it.')
    return False


def run_dataset(args, ds, dev):
    """Train each requested model once, then measure probes and geometry."""
    base_data = load_ts_dataset(ds, run_idx=0, device=dev)
    n = base_data['num_nodes']
    ei = base_data['edge_index']
    print(f"=== {ds}: n={n} classes={base_data['num_classes']}")
    entry = {
        'num_classes': base_data['num_classes'],
        'num_nodes': n,
        'paired_splits': bool(
            args.paired_splits and ds in MULTI_SPLIT_DATASETS),
        'models': {},
    }
    l0_dir = chunked_dirichlet(base_data['X'].float(), ei)
    for arch in args.architectures:
        entry['models'][arch] = {}
        for seed in args.seeds:
            use_paired = args.paired_splits and ds in MULTI_SPLIT_DATASETS
            run_idx = SEEDS.index(seed) if use_paired else 0
            data = load_ts_dataset(ds, run_idx=run_idx, device=dev)
            assert_split_only_change(base_data, data, ds, run_idx)
            set_seed(seed)
            model = build_and_train(arch, data, DEPTH, dev)
            probe = linear_probing(model, data['X'], data['labels'],
                                   data['train_mask'], data['test_mask'],
                                   data['num_classes'])
            model.eval()
            with torch.no_grad():
                _, hidden = model(data['X'], return_hidden=True)
            geom = measure_geometry(hidden, ei, input_dirichlet=l0_dir)
            entry['models'][arch][seed] = {
                'probe': probe,
                'geometry': geom,
                'split': split_metadata(data, run_idx),
            }
            del model, hidden
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  {ds} {arch} seed={seed} split={run_idx} trained")
    return entry


def main():
    args = parse_args()
    if args.device == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        dev = args.device
    out_pkl = os.path.join(args.out, 'geometry_grid_results.pkl')
    print(f"device: {dev}")
    print(f"paired splits: {args.paired_splits}")
    print(f"output: {out_pkl}")
    os.makedirs(args.out, exist_ok=True)
    results = load_results(args, out_pkl)
    for ds in args.datasets:
        if should_skip_dataset(args, ds, results):
            print(f"skip {ds} (done)")
            continue
        entry = run_dataset(args, ds, dev)
        results[ds] = entry
        save_results(results, out_pkl)
        arch0, seed0 = args.architectures[0], args.seeds[0]
        g4 = entry['models'][arch0][seed0]['geometry'][DEPTH]
        print(f"{ds} DONE {arch0} L4 DE={g4['dirichlet']:.6g} "
              f"global MAD={g4['mad_global']:.6g}")
    print("GRID DONE")


if __name__ == '__main__':
    main()
