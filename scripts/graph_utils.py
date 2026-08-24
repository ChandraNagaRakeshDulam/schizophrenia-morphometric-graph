import numpy as np
import torch
from torch_geometric.data import Data

def training_node_scaler(X_train):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std

def apply_node_scaler(X, mean, std):
    return (X - mean) / std

def adjacency_to_edges(A, density=0.20, rank_mode="absolute"):
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    vals = A[iu]
    if rank_mode == "absolute":
        rank = np.abs(vals)
    elif rank_mode == "positive":
        rank = vals.copy()
        rank[rank <= 0] = -np.inf
    else:
        raise ValueError(rank_mode)
    m = max(1, int(round(density * len(vals))))
    finite = np.isfinite(rank)
    if finite.sum() < m:
        m = int(finite.sum())
    if m == 0:
        raise ValueError("No eligible edges at requested density.")
    eligible = np.flatnonzero(finite)
    chosen_local = np.argpartition(rank[eligible], -m)[-m:]
    chosen = eligible[chosen_local]
    r, c = iu[0][chosen], iu[1][chosen]
    w = vals[chosen].astype(np.float32)
    edge_index = np.vstack([np.concatenate([r, c]), np.concatenate([c, r])])
    edge_attr = np.concatenate([w, w])[:, None]
    return torch.as_tensor(edge_index, dtype=torch.long), torch.as_tensor(edge_attr, dtype=torch.float32)

def make_graph(X, A, y, density, rank_mode="absolute", subject_index=None):
    edge_index, edge_attr = adjacency_to_edges(A, density, rank_mode)
    data = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([int(y)], dtype=torch.float32),
    )
    if subject_index is not None:
        data.subject_index = torch.tensor([int(subject_index)], dtype=torch.long)
    return data
