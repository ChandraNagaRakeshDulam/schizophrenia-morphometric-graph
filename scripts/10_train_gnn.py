#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GATv2Conv, global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import training_node_scaler, apply_node_scaler, make_graph

from models import GraphClassifier

def build_graphs(X, A, y, indices, density, rank_mode):
    return [make_graph(X[i], A[i], y[i], density, rank_mode, int(i)) for i in indices]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ss, idxs = [], [], []
    for b in loader:
        b = b.to(device)
        logits = model(b.x, b.edge_index, b.batch)
        ys.extend(b.y.view(-1).cpu().numpy().tolist())
        ss.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        idxs.extend(b.sample_idx.view(-1).cpu().numpy().tolist())
    auc = roc_auc_score(ys, ss) if len(set(ys)) > 1 else np.nan
    return auc, np.asarray(ys), np.asarray(ss), np.asarray(idxs)

def train_fixed_epochs(graphs, kind, hidden, epochs, seed, device):
    torch.manual_seed(seed)
    model = GraphClassifier(graphs[0].num_node_features, hidden, kind).to(device)
    loader = DataLoader(graphs, batch_size=min(16, len(graphs)), shuffle=True)
    ys = np.array([int(g.y.item()) for g in graphs])
    n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(int(epochs)):
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            logits = model(b.x, b.edge_index, b.batch)
            loss = loss_fn(logits, b.y.view(-1))
            loss.backward()
            opt.step()
    return model

def inner_score(X, A, y, train_idx, density, hidden, kind, rank_mode, seed, device):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs = []
    for k, (itr_rel, iva_rel) in enumerate(inner.split(train_idx, y[train_idx])):
        itr, iva = train_idx[itr_rel], train_idx[iva_rel]
        mean, std = training_node_scaler(X[itr])
        Xs = apply_node_scaler(X, mean, std)
        gtr = build_graphs(Xs, A, y, itr, density, rank_mode)
        gva = build_graphs(Xs, A, y, iva, density, rank_mode)
        # Fixed epoch count for hyperparameter comparison; tune this only within training data if expanded.
        model = train_fixed_epochs(gtr, kind, hidden, epochs=120, seed=seed+k, device=device)
        auc, *_ = evaluate(model, DataLoader(gva, batch_size=16), device)
        aucs.append(auc)
    return float(np.nanmean(aucs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--representation", choices=["msn","mind"], required=True)
    p.add_argument("--model", choices=["gcn","sage","gin","gat","gatv2"], required=True)
    p.add_argument("--densities", nargs="+", type=float, default=[0.10,0.20,0.30])
    p.add_argument("--hidden", nargs="+", type=int, default=[32,64])
    p.add_argument("--rank-mode", choices=["absolute","positive"], default="absolute")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    ids = d["ids"].astype(str)
    y = d["y"].astype(int)
    X = d["X_nodes"].astype(np.float32)
    A = d["A_mind" if a.representation == "mind" else "A_msn"].astype(np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=a.seed)
    pred_rows = []

    for split_id, (tr, te) in enumerate(outer.split(X, y)):
        repeat = split_id // 5
        fold = split_id % 5
        candidates = []
        for density in a.densities:
            for hidden in a.hidden:
                s = inner_score(X, A, y, tr, density, hidden, a.model, a.rank_mode, a.seed + split_id*100, device)
                candidates.append((s, density, hidden))
                print("split", split_id, "repeat", repeat, "fold", fold, "candidate", density, hidden, "inner_auc", s)
        candidates.sort(reverse=True, key=lambda z: z[0])
        best_auc, density, hidden = candidates[0]

        # Training-only scaling. Outer-test subjects do not influence mean/std.
        mean, std = training_node_scaler(X[tr])
        Xs = apply_node_scaler(X, mean, std)
        gtr = build_graphs(Xs, A, y, tr, density, a.rank_mode)
        gte = build_graphs(Xs, A, y, te, density, a.rank_mode)

        model = train_fixed_epochs(gtr, a.model, hidden, epochs=120, seed=a.seed + split_id, device=device)
        auc, yt, st, idxs = evaluate(model, DataLoader(gte, batch_size=16), device)

        ckpt = {
            "state_dict": model.state_dict(),
            "model_kind": a.model,
            "hidden": hidden,
            "density": density,
            "rank_mode": a.rank_mode,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "split": split_id,
            "repeat": repeat,
            "fold": fold,
            "test_indices": te,
            "inner_best_auc": best_auc,
        }
        torch.save(ckpt, out / f"split_{split_id}.pt")

        for yy, ss, ii in zip(yt, st, idxs):
            pred_rows.append({
                "split": split_id, "repeat": repeat, "fold": fold, "subject_index": int(ii), "subject_id": ids[ii],
                "y": int(yy), "score": float(ss), "density": density,
                "hidden": hidden, "inner_best_auc": best_auc,
            })

    pred = pd.DataFrame(pred_rows)
    pred.to_csv(out / "outer_predictions.csv", index=False)
    roc = roc_auc_score(pred.y, pred.score)
    pr = average_precision_score(pred.y, pred.score)
    metrics = {"roc_auc": float(roc), "pr_auc": float(pr), "device": str(device), "outer_cv": "5-fold x 3 repeats", "uses_edge_weights": False}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("NOTE: this starter nests density/hidden width, but final paper should also nest all optimizer/regularization/epoch choices or preregister them.")

if __name__ == "__main__":
    main()
