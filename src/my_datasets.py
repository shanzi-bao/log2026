"""Load the 12 multiclass benchmarks with the paper's split protocol."""

import os

import numpy as np
import torch
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
from torch_geometric.utils import to_undirected

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class HeterophilousDataset(object):
    """Base loader for the Platonov et al. (2023) heterophily benchmarks.

    These datasets ship with 10 official train/val/test splits; `split_idx`
    selects one. Edges are made undirected, matching the original paper.
    """
    NAME = None

    def __init__(self, split_idx=0):
        dataset = HeterophilousGraphDataset(root=f'/tmp/{self.NAME}', name=self.NAME)
        self.data = dataset[0]
        self.data.edge_index = to_undirected(self.data.edge_index)
        self.train_mask = self.data.train_mask[:, split_idx]
        self.valid_mask = self.data.val_mask[:, split_idx]
        self.test_mask = self.data.test_mask[:, split_idx]
        self.num_classes = dataset.num_classes


class AmazonRatingsDataset(HeterophilousDataset):
    NAME = 'Amazon-ratings'   # 24,492 nodes, 5 classes, moderate heterophily


class RomanEmpireDataset(HeterophilousDataset):
    NAME = 'Roman-empire'     # 22,662 nodes, 18 classes, strong heterophily


def class_rand_splits(label, label_num_per_class=20, valid_num=500,
                      test_num=1000, seed=123):
    """Random class-balanced split (ported from tunedGNN data_utils.py).

    label_num_per_class train nodes per class; valid_num / test_num sampled
    from the remaining nodes. Seeded generator so the split is reproducible.
    """
    g = torch.Generator().manual_seed(seed)
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    for c in label.squeeze().unique():
        idx_c = idx[label.squeeze() == c]
        perm = idx_c[torch.randperm(idx_c.shape[0], generator=g)]
        train_idx += perm[:label_num_per_class].tolist()
        non_train_idx += perm[label_num_per_class:].tolist()
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0], generator=g)]
    valid_idx = non_train_idx[:valid_num]
    test_idx = non_train_idx[valid_num:valid_num + test_num]
    return torch.as_tensor(train_idx), valid_idx, test_idx


def _idx_to_mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask


def _build_dict(x, y, edge_index, train_mask, valid_mask, test_mask, device):
    """Assemble the standard data dict (sparse adjacency only)."""
    from src.gcn import normalize_adj
    num_nodes = x.size(0)
    edge_index = edge_index.to(device)
    adj_norm = normalize_adj(edge_index=edge_index, num_nodes=num_nodes)
    return {
        'A': adj_norm,
        'adj_norm': adj_norm,
        'X': x.to(device),
        'labels': y.to(device),
        'train_mask': train_mask.to(device),
        'valid_mask': valid_mask.to(device),
        'test_mask': test_mask.to(device),
        'train_y': y[train_mask].to(device),
        'valid_y': y[valid_mask].to(device),
        'test_y': y[test_mask].to(device),
        'num_classes': int(y.max().item()) + 1,
        'num_nodes': num_nodes,
        'edge_index': edge_index,
    }


TS_DATASETS = ['cora', 'citeseer', 'pubmed',
               'amazon-photo', 'amazon-computer',
               'coauthor-cs', 'coauthor-physics', 'wikics',
               'squirrel', 'chameleon',
               'amazon-ratings', 'roman-empire']


def load_ts_dataset(name, run_idx=0, device='cpu'):
    """Load one of the 12 multiclass datasets with the tunedGNN split protocol.

    Split protocol per dataset family (mirrors tunedGNN main.py):
      - cora/citeseer/pubmed: ONE random class-balanced split (20 per class,
        500 val, 1000 test; fixed seed), model seed varies per run
      - amazon-*/coauthor-*: fixed 60/20/20 split shipped with the repo
        (data/<name>_split.npz), model seed varies per run
      - wikics: dataset-provided splits, split #run_idx (val = val|stopping)
      - roman-empire/amazon-ratings: official split #run_idx
      - squirrel/chameleon: Platonov FILTERED version, split #run_idx
    """
    from torch_geometric.utils import to_undirected

    name_l = name.lower().replace('_', '-')

    if name_l in ('cora', 'citeseer', 'pubmed'):
        # no feature normalization, matching tunedGNN's no_feat_norm=True
        ds = Planetoid(root=f'/tmp/{name_l.capitalize()}', name=name_l.capitalize())
        data = ds[0]
        tr, va, te = class_rand_splits(data.y)
        n = data.x.size(0)
        return _build_dict(data.x, data.y, data.edge_index,
                           _idx_to_mask(tr, n), _idx_to_mask(va, n),
                           _idx_to_mask(te, n), device)

    if name_l in ('amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics'):
        from torch_geometric.datasets import Amazon, Coauthor
        import torch_geometric.transforms as T
        if name_l.startswith('amazon'):
            pyg_name = 'Photo' if name_l == 'amazon-photo' else 'Computers'
            ds = Amazon(root='/tmp/Amazon', name=pyg_name, transform=T.NormalizeFeatures())
        else:
            pyg_name = 'CS' if name_l == 'coauthor-cs' else 'Physics'
            ds = Coauthor(root='/tmp/Coauthor', name=pyg_name, transform=T.NormalizeFeatures())
        data = ds[0]
        idx = np.load(os.path.join(DATA_DIR, f'{name_l}_split.npz'))
        n = data.x.size(0)
        return _build_dict(data.x, data.y, data.edge_index,
                           _idx_to_mask(torch.from_numpy(idx['train']), n),
                           _idx_to_mask(torch.from_numpy(idx['valid']), n),
                           _idx_to_mask(torch.from_numpy(idx['test']), n), device)

    if name_l == 'wikics':
        from torch_geometric.datasets import WikiCS
        ds = WikiCS(root='/tmp/wikics', is_undirected=True)
        data = ds[0]
        i = run_idx % data.train_mask.shape[1]
        valid_mask = data.val_mask[:, i] | data.stopping_mask[:, i]
        return _build_dict(data.x, data.y, data.edge_index,
                           data.train_mask[:, i], valid_mask, data.test_mask, device)

    if name_l in ('roman-empire', 'amazon-ratings'):
        cls = {'roman-empire': RomanEmpireDataset,
               'amazon-ratings': AmazonRatingsDataset}[name_l]
        d = cls(split_idx=run_idx % 10)
        return _build_dict(d.data.x, d.data.y, d.data.edge_index,
                           d.train_mask, d.valid_mask, d.test_mask, device)

    if name_l in ('squirrel', 'chameleon'):
        raw = np.load(os.path.join(DATA_DIR, f'{name_l}_filtered.npz'))
        x = torch.as_tensor(raw['node_features']).float()
        y = torch.as_tensor(raw['node_labels']).long()
        edge_index = to_undirected(torch.as_tensor(raw['edges']).T.long())
        i = run_idx % raw['train_masks'].shape[0]
        return _build_dict(x, y, edge_index,
                           torch.as_tensor(raw['train_masks'][i]),
                           torch.as_tensor(raw['val_masks'][i]),
                           torch.as_tensor(raw['test_masks'][i]), device)

    raise ValueError(f"Unknown TS dataset: {name}")
