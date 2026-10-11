"""Regression tests for smoothing trajectories and result loading."""

import argparse
import contextlib
import copy
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.q1 import run_smoothing_trajectories as experiment
from src.geometry_metrics import global_mean_average_distance


class FrozenModel:
    def __init__(self, hidden):
        self.hidden = hidden

    def eval(self):
        return self

    def __call__(self, x, return_hidden=False):
        return self.hidden[-1], self.hidden


class SmoothingTrajectoriesTests(unittest.TestCase):
    def setUp(self):
        x = torch.tensor([[1., 2.], [0., 3.], [-1., 1.]])
        self.hidden = [x + k / 10 for k in range(5)]
        self.edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.data = {
            'X': x, 'labels': torch.tensor([0, 1, 0]),
            'edge_index': self.edges, 'num_nodes': 3, 'num_classes': 2,
            'train_mask': torch.tensor([True, False, False]),
            'valid_mask': torch.tensor([False, True, False]),
            'test_mask': torch.tensor([False, False, True]),
        }
        self.probe = [
            {'layer': k, 'accuracy': 50., 'entropy_normalized': .5}
            for k in range(5)
        ]

    def arguments(self, directory, **overrides):
        values = dict(
            datasets=['cora'], architectures=['gcn'], seeds=[42, 123],
            out=directory, base_pkl=None, paired_splits=True,
            force=False, device='cpu',
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_combined_metrics_preserve_inputs_and_rng(self):
        before = [h.clone() for h in self.hidden]
        rng = torch.random.get_rng_state().clone()
        records = experiment.measure_geometry(self.hidden, self.edges)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for k, (h, record) in enumerate(zip(self.hidden, records)):
            self.assertTrue(torch.equal(h, before[k]))
            self.assertEqual(record['dirichlet'], experiment.chunked_dirichlet(h, self.edges))
            self.assertEqual(record['mad_global'], global_mean_average_distance(h))

    def test_training_once_per_seed_and_complete_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with patch.object(experiment, 'parse_args', return_value=args), \
                 patch.object(experiment, 'load_ts_dataset', return_value=self.data), \
                 patch.object(experiment, 'build_and_train', return_value=FrozenModel(self.hidden)) as train, \
                 patch.object(experiment, 'linear_probing', return_value=self.probe) as probe, \
                 contextlib.redirect_stdout(io.StringIO()):
                experiment.main()
                self.assertEqual(train.call_count, 2)
                self.assertEqual(probe.call_count, 2)
                path = Path(directory) / 'geometry_grid_results.pkl'
                saved = path.read_bytes()
                entry = pickle.loads(saved)['cora']
                self.assertTrue(experiment.entry_is_complete(entry, ['gcn'], [42, 123]))
                self.assertFalse(experiment.entry_is_complete(entry, ['gat'], [42]))
                experiment.main()
                self.assertEqual(train.call_count, 2)
                self.assertEqual(probe.call_count, 2)
                self.assertEqual(path.read_bytes(), saved)

    def test_incomplete_resume_and_base_archives_are_rejected(self):
        geometry = experiment.measure_geometry(self.hidden, self.edges)
        del geometry[2]['mad_global']
        entry = {'paired_splits': False, 'models': {'gcn': {
            42: {'probe': self.probe, 'geometry': geometry},
        }}}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'geometry_grid_results.pkl'
            for use_base in (False, True):
                for dataset in ('cora', 'citeseer'):
                    with self.subTest(use_base=use_base, dataset=dataset):
                        with archive.open('wb') as handle:
                            pickle.dump({dataset: entry}, handle)
                        args = self.arguments(
                            str(Path(directory) / 'new') if use_base else directory,
                            base_pkl=str(archive) if use_base else None,
                        )
                        with patch.object(experiment, 'parse_args', return_value=args), \
                             patch.object(experiment, 'build_and_train') as train, \
                             contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaisesRegex(ValueError, 'complete probe/DE/global-MAD'):
                                experiment.main()
                            train.assert_not_called()

    def test_combined_plot_data_matches_bundled_two_archive_route(self):
        from analysis.plots import smoothing

        combined, legacy, mad = {}, {}, {}
        values = experiment.measure_geometry(self.hidden, self.edges)
        for dataset, *_ in smoothing.MAIN_DATASETS:
            combined[dataset] = {'models': {'gcn': {42: {
                'probe': self.probe, 'geometry': copy.deepcopy(values),
            }}}}
            legacy[dataset] = copy.deepcopy(combined[dataset])
            for row in legacy[dataset]['models']['gcn'][42]['geometry']:
                del row['mad_global']
            mad[dataset] = {'models': {'gcn': {42: {
                'mad': [row['mad_global'] for row in values],
            }}}}
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('combined.pkl', 'legacy.pkl', 'mad.pkl')]
            for path, records in zip(paths, (combined, legacy, mad)):
                with path.open('wb') as handle:
                    pickle.dump(records, handle)
            with patch.object(smoothing, 'PKL', paths[1]), \
                 patch.object(smoothing, 'GLOBAL_MAD_PKL', paths[2]):
                self.assertEqual(smoothing.load_results(), combined)
            with patch.object(smoothing, 'GLOBAL_MAD_PKL', Path(directory) / 'absent.pkl'):
                self.assertEqual(smoothing.load_results(paths[0]), combined)
                with self.assertRaisesRegex(ValueError, 'combined DE/global-MAD'):
                    smoothing.load_results(paths[1])


if __name__ == '__main__':
    unittest.main()
