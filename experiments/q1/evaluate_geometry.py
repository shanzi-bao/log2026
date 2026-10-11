#!/usr/bin/env python3
"""Compute geometric diagnostics from saved GNN checkpoints.

Reuse the probe results associated with each checkpoint. Probe weights are
not needed. Verify the graph, split, and layer-wise group distance ratios,
then compute scale-normalized Dirichlet energy and global MAD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import torch

from src.gcn import SimpleGNN
from src.gat import GAT
from src.my_datasets import load_ts_dataset
from src.geometry_metrics import (
    dirichlet_energy,
    global_mean_average_distance,
    group_distance_ratio,
)

SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ['cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer',
            'coauthor-cs', 'coauthor-physics', 'wikics', 'amazon-ratings',
            'chameleon', 'squirrel', 'roman-empire']
MULTISPLIT = {'wikics', 'amazon-ratings', 'chameleon', 'squirrel', 'roman-empire'}
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True,
                        help='Directory containing supervised_control_results.pkl and checkpoints/')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/checkpoint_geometry')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--architectures', nargs='+', choices=['gcn', 'gat'], default=['gcn', 'gat'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rgroup-rtol', type=float, default=5e-4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    protected = [ROOT / 'results', ROOT / 'data', args.archive_dir]
    if any(args.output.resolve() == p.resolve() or args.output.resolve().is_relative_to(p.resolve())
           for p in protected):
        parser.error('--output must not overwrite code or archived reference/checkpoint files')
    return args


def prepare_evaluation(args):
    """Load probe results and record the evaluation settings and environment."""
    archive_path = args.archive_dir / 'supervised_control_results.pkl'
    with archive_path.open('rb') as handle:
        archive = pickle.load(handle)
    assert archive['configuration']['depth'] == 4
    assert archive['configuration']['paired_splits'] is True
    metadata = {
        'archive': str(archive_path),
        'configuration': archive['configuration'],
        'model_parameters': {'gcn_hidden_dim': 32, 'gat_hidden_dim_per_head': 8,
                             'gat_heads': 4, 'activation': 'relu'},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'device': 'cpu', 'threads': args.threads},
        'rgroup_verification': {'rtol': args.rgroup_rtol, 'atol': 1e-5},
    }
    metadata_path = args.output / 'metadata.json'
    if any((args.output / 'runs').glob('*.json')):
        if not metadata_path.exists():
            raise ValueError('Existing runs lack evaluation metadata; use a new --output directory')
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != metadata[key] for key in
               ('archive', 'configuration', 'model_parameters', 'environment', 'rgroup_verification')):
            raise ValueError('Existing runs use different settings or software; use a new --output directory')
    write_json(metadata_path, metadata)
    return archive


def measure_checkpoint(checkpoint, architecture, data, record,
                       input_geometry, input_rgroup, rgroup_rtol):
    """Verify saved R_Group values and measure geometry on the same model."""
    # The checkpoint replaces all initialized parameters.
    if architecture == 'gcn':
        model = SimpleGNN(data['X'].shape[1], 32, data['num_classes'], 4, data['adj_norm'])
    else:
        model = GAT(data['X'].shape[1], 8, data['num_classes'], 4, data['edge_index'], num_heads=4)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key, value in model.state_dict().items():
        if key.endswith('edge_index'):
            assert torch.equal(value, state[key]), f'Graph changed: {checkpoint}/{key}'
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry, differences = [], []
    with torch.no_grad():
        _, hidden = model(data['X'], return_hidden=True)
        assert len(hidden) == 5
        assert hidden[-1].shape == (data['num_nodes'], data['num_classes'])
        for layer, h in enumerate(hidden):
            if not torch.isfinite(h).all():
                raise ValueError(f'Non-finite representation {checkpoint}/L{layer}')
            rg = input_rgroup if layer == 0 else group_distance_ratio(h, data['labels'], data['train_mask'])
            saved = record['controls'][layer]['r_group']
            assert rg['num_classes'] == data['num_classes']
            for field in ['num_labelled_nodes', 'num_classes', 'min_class_size', 'max_class_size']:
                assert rg[field] == saved[field], (checkpoint, field)
            np.testing.assert_allclose(rg['ratio'], saved['ratio'], rtol=rgroup_rtol, atol=1e-5,
                                       err_msg=f'R_Group mismatch {checkpoint}/L{layer}')
            differences.append(abs(rg['ratio'] - saved['ratio']) / max(abs(saved['ratio']), 1e-12))
            metrics = input_geometry if layer == 0 else {
                'dirichlet': dirichlet_energy(h, data['edge_index']),
                'mad': global_mean_average_distance(h),
            }
            geometry.append({'layer': layer, **metrics, 'r_group': rg['ratio'],
                             'saved_r_group': saved['ratio']})
    return geometry, differences


def result_matches(saved, identity, record):
    """Check that a completed output still matches this checkpoint and archive run."""
    if any(saved.get(key) != value for key, value in identity.items()):
        return False
    if saved.get('probe') != record['probe']:
        return False
    geometry = saved.get('geometry', [])
    if [row.get('layer') for row in geometry] != list(range(5)):
        return False
    for row, control in zip(geometry, record['controls']):
        if row.get('saved_r_group') != control['r_group']['ratio']:
            return False
        for metric in ('dirichlet', 'mad', 'r_group'):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                return False
    checks = saved.get('verification', {})
    return all(checks.get(key) is True for key in
               ('all_layers_rgroup_match', 'strict_state_dict_loaded'))


def evaluate_dataset(args, dataset, archive):
    """Evaluate each checkpoint, checking splits and saving after every model."""
    started_dataset = time.time()
    print(f'Loading {dataset}', flush=True)
    # Only one large feature tensor at a time; split masks change per seed.
    data = load_ts_dataset(dataset, run_idx=0, device='cpu')
    current_split = 0
    entry = archive['datasets'][dataset]
    assert data['num_nodes'] == entry['num_nodes']
    assert data['num_classes'] == entry['num_classes']
    input_geometry = {
        'dirichlet': dirichlet_energy(data['X'], data['edge_index']),
        'mad': global_mean_average_distance(data['X']),
    }
    input_rgroup_cache = {}
    for seed in args.seeds:
        run_idx = SEEDS.index(seed) if dataset in MULTISPLIT else 0
        if current_split != run_idx:
            new_data = load_ts_dataset(dataset, run_idx=run_idx, device='cpu')
            for key in ('num_nodes', 'num_classes'):
                assert new_data[key] == data[key], f'{dataset}: {key} changed across splits'
            for key in ('edge_index', 'X', 'labels'):
                assert torch.equal(new_data[key], data[key]), f'{dataset}: {key} changed across splits'
            data = new_data
            del new_data
            current_split = run_idx
        split = {'run_idx': run_idx,
                 **{f'{part}_nodes': int(data[f'{part}_mask'].sum()) for part in ['train', 'valid', 'test']}}
        if run_idx not in input_rgroup_cache:
            input_rgroup_cache[run_idx] = group_distance_ratio(data['X'], data['labels'], data['train_mask'])
        for architecture in args.architectures:
            started = time.time()
            checkpoint = args.archive_dir / 'checkpoints' / f'{dataset}_{architecture}_depth4_seed{seed}.pt'
            checkpoint_hash = sha256(checkpoint)
            path = args.output / 'runs' / f'{dataset}_{architecture}_seed{seed}.json'
            record = entry['models'][architecture][seed]
            assert record['seed'] == seed and record['split'] == split
            assert [p['layer'] for p in record['probe']] == list(range(5))
            assert [p['layer'] for p in record['controls']] == list(range(5))
            for control in record['controls']:
                for field in ('num_labelled_nodes', 'num_classes', 'min_class_size', 'max_class_size'):
                    assert control['r_group'][field] == input_rgroup_cache[run_idx][field], (checkpoint, field)
            identity = {'dataset': dataset, 'architecture': architecture, 'seed': seed,
                        'split': split, 'checkpoint_sha256': checkpoint_hash}
            if path.exists() and not args.force:
                old = json.loads(path.read_text())
                if not result_matches(old, identity, record):
                    raise ValueError(f'Existing result does not match the checkpoint or archive: {path}; '
                                     'use --force to recompute it')
                print(f'Skip {dataset} {architecture} {seed}: already verified', flush=True)
                continue
            geometry, differences = measure_checkpoint(
                checkpoint, architecture, data, record, input_geometry,
                input_rgroup_cache[run_idx], args.rgroup_rtol,
            )
            output = {
                **identity, 'checkpoint': str(checkpoint),
                'probe': record['probe'], 'geometry': geometry,
                'verification': {'all_layers_rgroup_match': True, 'max_rgroup_relative_error': max(differences),
                                 'rgroup_relative_errors': differences,
                                 'strict_state_dict_loaded': True, 'all_train_classes_present': True,
                                 'probe_source': 'unmodified scores from this same saved GNN experiment'},
                'elapsed_seconds': time.time() - started,
            }
            write_json(path, output)
            print(f'Done {dataset} {architecture} {seed}: max R_Group relative error={max(differences):.3g}; {time.time()-started:.1f}s', flush=True)
    print(f'Finished {dataset} in {time.time()-started_dataset:.1f}s', flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    archive = prepare_evaluation(args)
    for dataset in args.datasets:
        evaluate_dataset(args, dataset, archive)


if __name__ == '__main__':
    main()
