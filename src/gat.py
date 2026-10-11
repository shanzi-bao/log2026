"""GAT layers and full-batch training utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils import add_self_loops

NUM_EPOCHS = 100
LR = 0.01


def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


# ============================================================
# Model
# ============================================================

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_index, num_heads=1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat

        self.register_buffer('edge_index', edge_index)

        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, h):
        N = h.size(0)
        src, dst = self.edge_index

        h = self.W(h).view(N, self.num_heads, self.out_features)

        e_src = (h[src] * self.a_src).sum(dim=-1)
        e_dst = (h[dst] * self.a_dst).sum(dim=-1)
        e = F.leaky_relu(e_src + e_dst, 0.2)

        e_max = torch.zeros(N, self.num_heads, device=h.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=False)
        e = torch.exp(e - e_max[dst])

        e_sum = torch.zeros(N, self.num_heads, device=h.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        alpha = e / (e_sum[dst] + 1e-16)

        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.num_heads, self.out_features, device=h.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        if self.concat:
            return out.view(N, self.num_heads * self.out_features)
        else:
            return out.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gat_layers, edge_index, num_heads=4):
        super().__init__()
        self.num_gat_layers = num_gat_layers

        edge_index, _ = add_self_loops(edge_index, num_nodes=None)

        self.layers = nn.ModuleList()

        if num_gat_layers == 1:
            self.layers.append(GATLayer(input_dim, output_dim, edge_index,
                                         num_heads=1, concat=False))
        else:
            self.layers.append(GATLayer(input_dim, hidden_dim, edge_index,
                                         num_heads=num_heads, concat=True))
            for _ in range(num_gat_layers - 2):
                self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, edge_index,
                                             num_heads=num_heads, concat=True))
            self.layers.append(GATLayer(hidden_dim * num_heads, output_dim, edge_index,
                                         num_heads=1, concat=False))

    def forward(self, x, return_hidden=False):
        hidden_states = [x] if return_hidden else None

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
            if return_hidden:
                hidden_states.append(h)

        if return_hidden:
            return h, hidden_states
        return h


# ============================================================
# Training utilities
# ============================================================

def train_gat(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.detach().item()


def evaluate_gat(X, y, mask, model):
    model.eval()
    with torch.no_grad():
        y_hat = model(X)[mask]
        y_hat = y_hat.max(1)[1]
        num_correct = y_hat.eq(y).sum()
        num_total = len(y)
        accuracy = 100.0 * (num_correct / num_total)
    return accuracy.item()


def train_eval_loop_gat(model, train_x, train_y, train_mask,
                        valid_x, valid_y, valid_mask,
                        test_x, test_y, test_mask):
    optimiser = Adam(model.parameters(), lr=LR)
    training_stats = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_gat(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gat(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gat(valid_x, valid_y, valid_mask, model)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} "
                  f"train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")

        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch': epoch}
        training_stats = update_stats(training_stats, epoch_stats)

    test_acc = evaluate_gat(test_x, test_y, test_mask, model)
    print(f"Test accuracy: {test_acc:.3f}")
    return training_stats
