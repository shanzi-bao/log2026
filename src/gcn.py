"""GCN layers and full-batch training utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Linear, Module, ModuleList
NUM_EPOCHS =  100
LR         = 0.01

def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats

def normalize_adj(A=None, edge_index=None, num_nodes=None, sparse=True):
    """Compute the symmetrically normalized adjacency D~^{-1/2}(A+I)D~^{-1/2} once.

    Pass either a raw dense adjacency `A` (unweighted) or an `edge_index`
    (with `num_nodes`); edges are assumed undirected (both directions present).
    Returns a torch.sparse_coo_tensor by default (dense if sparse=False).

    Compute once per graph and share across GCN layers to avoid storing
    duplicate adjacency matrices.
    """
    if edge_index is None:
        if A is None:
            raise ValueError("pass either A or edge_index")
        num_nodes = A.size(0)
        edge_index = A.nonzero().t()
    if num_nodes is None:
        raise ValueError("num_nodes is required with edge_index")
    device = edge_index.device

    # A_tilde = A + I
    loop = torch.arange(num_nodes, device=device)
    src = torch.cat([edge_index[0], loop])
    dst = torch.cat([edge_index[1], loop])

    # D_tilde^{-1/2}
    ones = torch.ones(src.numel(), device=device)
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, dst, ones)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[dst] * deg_inv_sqrt[src]
    adj_norm = torch.sparse_coo_tensor(
        torch.stack([dst, src]), values, (num_nodes, num_nodes)
    ).coalesce()
    return adj_norm if sparse else adj_norm.to_dense()


class GCNLayer(Module):
    """Graph Convolutional Network layer from Kipf & Welling.

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output feature vectors
        A (torch.Tensor): either a raw dense adjacency matrix (normalized
            within the layer), or a precomputed normalized
            adjacency from normalize_adj() — sparse, or dense with
            precomputed=True. Prefer passing one shared precomputed matrix
            to every layer.
        precomputed (bool): set True when A is already normalized
    """
    def __init__(self, input_dim, output_dim, A, precomputed=False):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if precomputed or A.is_sparse:
            self.adj_norm = A
        else:
            self.adj_norm = normalize_adj(A=A, sparse=False)

        # Linear transformation
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x):
        """Implements the forward pass for the layer

        Args:
            x (torch.Tensor): input node feature matrix
        """
        # H = sigma(D^{-1/2} A_tilde D^{-1/2} X W)
        if self.adj_norm.is_sparse:
            x = torch.sparse.mm(self.adj_norm, x)
        else:
            x = self.adj_norm @ x
        x = self.linear(x)

        return x
    
# Training and evaluation

def train_gnn_cora(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data


def evaluate_gnn_cora(X, y, mask, model):
    model.eval()
    with torch.no_grad():
        y_hat = model(X)[mask]
        y_hat = y_hat.data.max(1)[1]
        num_correct = y_hat.eq(y.data).sum()
        num_total = len(y)
        accuracy = 100.0 * (num_correct/num_total)
    return accuracy


# Training loop
def train_eval_loop_gnn_cora(model, train_x, train_y, train_mask,
                        valid_x, valid_y, valid_mask,
                        test_x, test_y, test_mask
                    ):
    optimiser = Adam(model.parameters(), lr=LR)
    training_stats = None
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_gnn_cora(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gnn_cora(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gnn_cora(valid_x, valid_y, valid_mask, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Evaluate on the test split after training.
    test_acc = evaluate_gnn_cora(test_x, test_y, test_mask, model)
    print(f"Test accuracy: {test_acc:.3f}")
    return training_stats

class SimpleGNN(Module):
    """A Simple GNN model using the GCNLayer for node classification

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Number of classes
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, A):
        super(SimpleGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Normalize once and share across all layers. A may be a raw dense
        # adjacency (normalized here) or an already-normalized sparse matrix
        # from normalize_adj() / a dataset loader.
        adj_norm = A if A.is_sparse else normalize_adj(A=A, sparse=True)
        self.adj_norm = adj_norm

        if num_gcn_layers > 1:
          self.gcn_layers = [GCNLayer(input_dim, hidden_dim, adj_norm, precomputed=True)]
          self.gcn_layers += [GCNLayer(hidden_dim, hidden_dim, adj_norm, precomputed=True) for i in range(num_gcn_layers-2)]
          self.gcn_layers += [GCNLayer(hidden_dim, output_dim, adj_norm, precomputed=True)]
        else:
          self.gcn_layers = [GCNLayer(input_dim, output_dim, adj_norm, precomputed=True)]

        self.gcn_layers = ModuleList(self.gcn_layers)
        self.num_gcn_layers = num_gcn_layers

    def forward(self, x, return_hidden=False):
        hidden_states = [x]  # h^(0) = original input

        for j in range(self.num_gcn_layers-1):
            x = self.gcn_layers[j](x)
            x = F.relu(x)
            hidden_states.append(x)  # h^(1), h^(2), 

        x = self.gcn_layers[-1](x)
        hidden_states.append(x)  # h^(K)

        if return_hidden:
            return x, hidden_states
        return x
