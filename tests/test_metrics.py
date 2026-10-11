"""Small deterministic metric tests.

Run from the repository root: python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.geometry_metrics import (
    dirichlet_energy,
    global_mean_average_distance,
    group_distance_ratio,
)


def brute_force_r_group(x, y):
    """Pairwise class-balanced squared distances after row normalization."""
    x = F.normalize(x.float(), p=2, dim=1, eps=1e-12)
    groups = [x[y == class_id] for class_id in torch.unique(y, sorted=True)]
    intra = []
    for group in groups:
        distances = (group[:, None, :] - group[None, :, :]).square().sum(dim=2)
        intra.append(distances.mean())
    inter = []
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            distances = (
                groups[i][:, None, :] - groups[j][None, :, :]
            ).square().sum(dim=2)
            inter.append(distances.mean())
    intra_value = torch.stack(intra).mean().item()
    inter_value = torch.stack(inter).mean().item()
    return inter_value / intra_value, inter_value, intra_value


class MetricTests(unittest.TestCase):
    def test_group_distance_ratio_matches_pairwise_computation(self):
        x = torch.tensor([
            [1.0, 0.0, 0.2], [0.8, 0.1, 0.3],
            [-1.0, 0.0, 0.1], [-0.7, -0.2, 0.4],
            [0.0, 1.0, -0.2], [0.2, 0.8, -0.1],
        ])
        y = torch.tensor([0, 0, 1, 1, 2, 2])
        mask = torch.ones(len(y), dtype=torch.bool)
        expected = brute_force_r_group(x, y)
        actual = group_distance_ratio(x, y, mask)
        np.testing.assert_allclose(
            [actual["ratio"], actual["inter_class_distance"],
             actual["intra_class_distance"]],
            expected, rtol=1e-5, atol=1e-6,
        )

    def test_group_distance_ratio_uses_training_nodes_only(self):
        x = torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0],
                          [-0.8, 0.2], [5.0, 9.0]])
        y = torch.tensor([0, 0, 1, 1, 2])
        mask = torch.tensor([True, True, True, True, False])
        actual = group_distance_ratio(x, y, mask)
        np.testing.assert_allclose(
            actual["ratio"], brute_force_r_group(x[mask], y[mask])[0],
            rtol=1e-5,
        )
        self.assertEqual(actual["num_classes"], 2)
        self.assertEqual(actual["num_labelled_nodes"], 4)

    def test_dirichlet_matches_explicit_degree_normalization(self):
        h = torch.tensor([[1.0, 2.0], [0.0, 3.0], [-1.0, 1.0]])
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        normalized = h / torch.tensor([2.0, 3.0, 2.0]).sqrt()[:, None]
        squared = (normalized[edges[0]] - normalized[edges[1]]).square().sum(1)
        expected = squared.mean() / h.square().sum(1).mean()
        self.assertAlmostEqual(dirichlet_energy(h, edges), float(expected), places=6)
        self.assertAlmostEqual(
            dirichlet_energy(7 * h, edges), float(expected), places=6,
        )

    def test_global_mad_matches_distinct_pairs_with_zero_rows(self):
        h = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
        hn = F.normalize(h, dim=1, eps=1e-12)
        distance = 1 - hn @ hn.T
        mask = ~torch.eye(len(h), dtype=torch.bool)
        expected = float(distance[mask].mean())
        self.assertAlmostEqual(global_mean_average_distance(h), expected, places=6)
        self.assertEqual(global_mean_average_distance(h[:1]), 0.0)

if __name__ == "__main__":
    unittest.main()
