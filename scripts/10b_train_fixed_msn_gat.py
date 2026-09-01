#!/usr/bin/env python3

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import training_node_scaler, apply_node_scaler
from models import GraphClassifier


def consensus_edges(A_train, density):
    """
    Build one shared topology using TRAINING SUBJECTS ONLY.
    Rank edges by mean absolute MSN magnitude.
    """
    n = A_train.shape[1]
    iu = np.triu_indices(n, k=1)

    strength = np.mean(np.abs(A_train[:, iu[0], iu[1]]), axis=0)

    m = max(1, int(round(density * len(strength))))
    chosen = np.argpartition(strength, -m)[-m:]

    r = iu[0][chosen]
    c = iu[1][chosen]

    edge_index = np.vstack([
        np.concatenate([r, c]),
        np.concatenate([c, r])
    ])

    return torch.as_tensor(edge_index, dtype=torch.long)


def build_graphs(X, y, indices, edge_index):
    graphs = []

    for i in indices:
        g = Data(
            x=torch.as_tensor(X[i], dtype=torch.float32),
            edge_index=edge_index.clone(),
            y=torch.tensor([int(y[i])], dtype=torch.float32),
        )

        g.sample_idx = torch.tensor([int(i)], dtype=torch.long)
        graphs.append(g)

    return graphs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ss, idxs = [], [], []

    for b in loader:
        b = b.to(device)
        logits = model(b.x, b.edge_index, b.batch)

        ys.extend(b.y.view(-1).cpu().numpy())
        ss.extend(torch.sigmoid(logits).cpu().numpy())
        idxs.extend(b.sample_idx.view(-1).cpu().numpy())

    return (
        roc_auc_score(ys, ss),
        np.asarray(ys),
        np.asarray(ss),
        np.asarray(idxs)
    )


def train_model(graphs, hidden, epochs, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GraphClassifier(
        graphs[0].num_node_features,
        hidden,
        "gat"
    ).to(device)

    loader = DataLoader(
        graphs,
        batch_size=min(16, len(graphs)),
        shuffle=True
    )

    ys = np.array([int(g.y.item()) for g in graphs])

    n_pos = ys.sum()
    n_neg = len(ys) - n_pos

    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        dtype=torch.float32,
        device=device
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    for _ in range(epochs):
        model.train()

        for b in loader:
            b = b.to(device)

            opt.zero_grad()
            logits = model(b.x, b.edge_index, b.batch)

            loss = loss_fn(logits, b.y.view(-1))
            loss.backward()
            opt.step()

    return model


def inner_score(X, A, y, tr, density, hidden, seed, device):

    inner = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=seed
    )

    aucs = []

    for k, (itr_rel, iva_rel) in enumerate(
        inner.split(tr, y[tr])
    ):
        itr = tr[itr_rel]
        iva = tr[iva_rel]

        # Training-only feature normalization
        mean, std = training_node_scaler(X[itr])
        Xs = apply_node_scaler(X, mean, std)

        # Inner-training-only consensus graph
        edge_index = consensus_edges(A[itr], density)

        gtr = build_graphs(Xs, y, itr, edge_index)
        gva = build_graphs(Xs, y, iva, edge_index)

        model = train_model(
            gtr,
            hidden,
            120,
            seed + k,
            device
        )

        auc, *_ = evaluate(
            model,
            DataLoader(gva, batch_size=16),
            device
        )

        aucs.append(auc)

    return float(np.mean(aucs))


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--densities", nargs="+", type=float,
                   default=[0.10, 0.20, 0.30])
    p.add_argument("--hidden", nargs="+", type=int,
                   default=[32, 64])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    d = np.load(a.dataset, allow_pickle=True)

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)
    X = d["X_nodes"].astype(np.float32)
    A = d["A_msn"].astype(np.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=a.seed
    )

    rows = []

    for split_id, (tr, te) in enumerate(outer.split(X, y)):

        repeat = split_id // 5
        fold = split_id % 5

        candidates = []

        for density in a.densities:
            for hidden in a.hidden:

                score = inner_score(
                    X, A, y, tr,
                    density, hidden,
                    a.seed + split_id * 100,
                    device
                )

                candidates.append((score, density, hidden))

                print(
                    "split", split_id,
                    "repeat", repeat,
                    "fold", fold,
                    "density", density,
                    "hidden", hidden,
                    "inner_auc", score
                )

        candidates.sort(reverse=True, key=lambda x: x[0])

        best_auc, density, hidden = candidates[0]

        mean, std = training_node_scaler(X[tr])
        Xs = apply_node_scaler(X, mean, std)

        # OUTER-TRAINING-only shared topology
        edge_index = consensus_edges(A[tr], density)

        gtr = build_graphs(Xs, y, tr, edge_index)
        gte = build_graphs(Xs, y, te, edge_index)

        model = train_model(
            gtr,
            hidden,
            120,
            a.seed + split_id,
            device
        )

        _, yt, st, idxs = evaluate(
            model,
            DataLoader(gte, batch_size=16),
            device
        )

        for yy, ss, ii in zip(yt, st, idxs):
            rows.append({
                "split": split_id,
                "repeat": repeat,
                "fold": fold,
                "subject_index": int(ii),
                "subject_id": ids[ii],
                "y": int(yy),
                "score": float(ss),
                "density": density,
                "hidden": hidden,
                "inner_best_auc": best_auc
            })

    pred = pd.DataFrame(rows)
    pred.to_csv(out / "outer_predictions.csv", index=False)

    metrics = {
        "roc_auc": float(roc_auc_score(pred.y, pred.score)),
        "pr_auc": float(average_precision_score(pred.y, pred.score)),
        "device": str(device),
        "outer_cv": "5-fold x 3 repeats",
        "topology": "training_consensus_msn",
        "subject_specific_edges": False
    }

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
