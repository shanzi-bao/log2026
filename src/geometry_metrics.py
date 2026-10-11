"""Geometric diagnostics on fixed node representations.

Dirichlet energy and global MAD are label-free. R_Group follows the authors'
DGN implementation (Zhou et al., NeurIPS 2020): node representations are
row-normalized, and squared Euclidean distances are averaged equally over
classes and class pairs. It is evaluated on labelled training nodes only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _edge_chunk_size(h, max_elements=4_000_000):
    """Choose an edge chunk that bounds temporary edge-feature tensors."""
    return max(1, max_elements // max(1, h.size(1)))


def dirichlet_energy(h, edge_index):
    """Mean over edges of ||h_u/sqrt(1+d_u) - h_v/sqrt(1+d_v)||^2,
    divided by the mean squared node norm (scale-invariant)."""
    src, dst = edge_index
    deg = torch.zeros(h.size(0), device=h.device).scatter_add_(
        0, dst, torch.ones(dst.numel(), device=h.device))
    inv_sqrt_deg = (1.0 + deg).rsqrt()
    total = h.new_zeros(())
    chunk_size = _edge_chunk_size(h)
    for start in range(0, src.numel(), chunk_size):
        end = min(start + chunk_size, src.numel())
        src_chunk = src[start:end]
        dst_chunk = dst[start:end]
        diff = (
            h[src_chunk] * inv_sqrt_deg[src_chunk].unsqueeze(1)
            - h[dst_chunk] * inv_sqrt_deg[dst_chunk].unsqueeze(1)
        )
        total += diff.square().sum()
    energy = total / src.numel()
    scale = h.pow(2).sum(dim=1).mean().clamp_min(1e-12)
    return (energy / scale).item()


def global_mean_average_distance(h):
    """Global MAD: mean cosine distance over distinct unordered node pairs.

    This is the all-pairs target-mask case of Chen et al. (2020). The
    algebraic form avoids materializing the dense node-by-node distance
    matrix. Zero-norm rows are normalized to zero using ``eps=1e-12``.
    """
    hn = F.normalize(h, dim=1, eps=1e-12)
    n = hn.size(0)
    if n < 2:
        return 0.0
    total_cosine = hn.sum(dim=0).pow(2).sum()
    diagonal_cosine = hn.pow(2).sum()
    mean_cosine = (total_cosine - diagonal_cosine) / (n * (n - 1))
    return (1.0 - mean_cosine).item()


@torch.no_grad()
def group_distance_ratio(
    representations: torch.Tensor,
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Compute the training-only Group Distance Ratio exactly and in O(nd).

    The official implementation explicitly constructs every within-class and
    between-class squared-distance matrix.  For groups X and Y, the same mean
    squared distance can be written as

        E||X-Y||^2 = E||X||^2 + E||Y||^2 - 2 E[X]^T E[Y].

    This identity avoids an O(n^2) matrix while preserving the official class-
    balanced averaging: each class contributes equally to the intra-class
    term and each unordered class pair contributes equally to the inter-class
    term, irrespective of class size.
    """
    if representations.ndim != 2:
        raise ValueError("representations must have shape [num_nodes, dim]")
    if labels.ndim != 1 or labelled_mask.ndim != 1:
        raise ValueError("labels and labelled_mask must be one-dimensional")
    if representations.shape[0] != labels.numel() or labels.numel() != labelled_mask.numel():
        raise ValueError("representations, labels, and mask must describe the same nodes")
    if labelled_mask.dtype != torch.bool:
        raise TypeError("labelled_mask must be a boolean tensor")

    x = F.normalize(
        representations[labelled_mask].detach().float(),
        p=2,
        dim=1,
        eps=eps,
    )
    y = labels[labelled_mask].long()
    if x.shape[0] == 0:
        raise ValueError("labelled_mask selects no nodes")

    classes = torch.unique(y, sorted=True)
    if classes.numel() < 2:
        raise ValueError("R_Group requires at least two represented classes")

    means: list[torch.Tensor] = []
    second_moments: list[torch.Tensor] = []
    within_distances: list[torch.Tensor] = []
    class_sizes: list[int] = []
    for class_id in classes:
        class_x = x[y == class_id]
        if class_x.shape[0] == 0:  # Defensive; torch.unique makes this unreachable.
            continue
        mean = class_x.mean(dim=0)
        second_moment = class_x.square().sum(dim=1).mean()
        # Mean over all ordered pairs, including the zero diagonal, matching
        # np.mean(pairwise_squared_distance_matrix) in the official code.
        within = 2.0 * (second_moment - mean.square().sum())
        means.append(mean)
        second_moments.append(second_moment)
        within_distances.append(within.clamp_min(0.0))
        class_sizes.append(int(class_x.shape[0]))

    intra = torch.stack(within_distances).mean()
    between_distances: list[torch.Tensor] = []
    for i in range(len(means) - 1):
        for j in range(i + 1, len(means)):
            between = (
                second_moments[i]
                + second_moments[j]
                - 2.0 * torch.dot(means[i], means[j])
            )
            between_distances.append(between.clamp_min(0.0))
    inter = torch.stack(between_distances).mean()

    intra_value = float(intra.item())
    inter_value = float(inter.item())
    # The DGN code assigns 1 to the 0/0 complete-collapse case.  An exactly
    # zero denominator with positive inter-class distance is retained as +inf.
    # The analysis script rejects non-finite endpoint values.
    if intra_value <= eps and inter_value <= eps:
        ratio = 1.0
    elif intra_value <= eps:
        ratio = float("inf")
    else:
        ratio = inter_value / intra_value

    return {
        "ratio": ratio,
        "inter_class_distance": inter_value,
        "intra_class_distance": intra_value,
        "num_labelled_nodes": int(x.shape[0]),
        "num_classes": int(classes.numel()),
        "min_class_size": min(class_sizes),
        "max_class_size": max(class_sizes),
    }
