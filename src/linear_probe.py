import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_probing(model, x, y_true, train_mask, test_mask, num_classes):
    model.eval()
    with torch.no_grad():
        _, hidden_states = model(x, return_hidden=True)

    results = []

    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()

        # Create probe
        probe = nn.Linear(h_k.shape[1], num_classes).to(h_k.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

        # Train probe
        for epoch in range(100):
            logits = probe(h_k[train_mask])
            loss = F.cross_entropy(logits, y_true[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate probe
        with torch.no_grad():
            probs = F.softmax(probe(h_k[test_mask]), dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            preds = probs.argmax(dim=1)
            accuracy = (preds == y_true[test_mask]).float().mean().item() * 100

        result = {
            'layer': k,
            'entropy': entropy.item(),
            'entropy_normalized': entropy.item() / math.log(num_classes),
            'accuracy': accuracy
        }
        results.append(result)
        print(f"Layer {k}: entropy = {entropy:.4f}, accuracy = {accuracy:.2f}%")

    return results


def linear_probing_per_node(model, x, y_true, train_mask, test_mask, num_classes):
    """Fit independent probes and store per-node entropy and true-class NLL."""
    model.eval()
    with torch.no_grad():
        _, hidden_states = model(x, return_hidden=True)

    # Train all probes
    probes = []
    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()
        probe = nn.Linear(h_k.shape[1], num_classes).to(h_k.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

        for epoch in range(100):
            logits = probe(h_k[train_mask])
            loss = F.cross_entropy(logits, y_true[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probes.append(probe)

    results = []

    for k, h_k in enumerate(hidden_states):
        h_k = h_k.detach()

        with torch.no_grad():
            probs = F.softmax(probes[k](h_k[test_mask]), dim=1)
            node_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

            p_true = probs[range(len(y_true[test_mask])), y_true[test_mask]]
            neg_log_p_true = -torch.log(p_true + 1e-8)

        results.append({
            'layer': k,
            'num_classes': num_classes,   # for post-hoc H / ln C normalization
            'entropy_per_node': node_entropy.cpu().numpy(),
            'neg_log_p_true_per_node': neg_log_p_true.cpu().numpy(),
        })

    return results
