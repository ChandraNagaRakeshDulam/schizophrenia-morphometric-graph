#!/usr/bin/env python3

import argparse
import json
import sys
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


def make_xonly_graph(X, y, subject_index):
    # No inter-regional edges.
    # PyG GATConv will internally add self-loops.
    edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=torch.as_tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([int(y)], dtype=torch.float32),
    )

    data.sample_idx = torch.tensor(
        [int(subject_index)],
        dtype=torch.long
    )

    return data


def build_graphs(X, y, indices):
    return [
        make_xonly_graph(X[i], y[i], int(i))
        for i in indices
    ]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    ys = []
    scores = []
    idxs = []

    for b in loader:
        b = b.to(device)

        logits = model(
            b.x,
            b.edge_index,
            b.batch
        )

        ys.extend(
            b.y.view(-1).cpu().numpy().tolist()
        )

        scores.extend(
            torch.sigmoid(logits).cpu().numpy().tolist()
        )

        idxs.extend(
            b.sample_idx.view(-1).cpu().numpy().tolist()
        )

    auc = (
        roc_auc_score(ys, scores)
        if len(set(ys)) > 1
        else np.nan
    )

    return (
        auc,
        np.asarray(ys),
        np.asarray(scores),
        np.asarray(idxs),
    )


def train_fixed_epochs(
    graphs,
    kind,
    hidden,
    epochs,
    seed,
    device
):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GraphClassifier(
        graphs[0].num_node_features,
        hidden,
        kind
    ).to(device)

    loader = DataLoader(
        graphs,
        batch_size=min(16, len(graphs)),
        shuffle=True
    )

    ys = np.array([
        int(g.y.item())
        for g in graphs
    ])

    n_pos = ys.sum()
    n_neg = len(ys) - n_pos

    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        device=device,
        dtype=torch.float32
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    for _ in range(int(epochs)):
        model.train()

        for b in loader:
            b = b.to(device)

            opt.zero_grad()

            logits = model(
                b.x,
                b.edge_index,
                b.batch
            )

            loss = loss_fn(
                logits,
                b.y.view(-1)
            )

            loss.backward()
            opt.step()

    return model


def inner_score(
    X,
    y,
    train_idx,
    hidden,
    kind,
    seed,
    device
):
    inner = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=seed
    )

    aucs = []

    for k, (itr_rel, iva_rel) in enumerate(
        inner.split(train_idx, y[train_idx])
    ):
        itr = train_idx[itr_rel]
        iva = train_idx[iva_rel]

        # Training-only normalization
        mean, std = training_node_scaler(X[itr])

        Xs = apply_node_scaler(
            X,
            mean,
            std
        )

        gtr = build_graphs(
            Xs,
            y,
            itr
        )

        gva = build_graphs(
            Xs,
            y,
            iva
        )

        model = train_fixed_epochs(
            gtr,
            kind,
            hidden,
            epochs=120,
            seed=seed + k,
            device=device
        )

        auc, *_ = evaluate(
            model,
            DataLoader(gva, batch_size=16),
            device
        )

        aucs.append(auc)

    return float(np.nanmean(aucs))


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True
    )

    p.add_argument(
        "--model",
        choices=[
            "gcn",
            "sage",
            "gin",
            "gat",
            "gatv2"
        ],
        default="gat"
    )

    p.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[32, 64]
    )

    p.add_argument(
        "--out",
        required=True
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    a = p.parse_args()

    d = np.load(
        a.dataset,
        allow_pickle=True
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(
        np.float32
    )

    out = Path(a.out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=a.seed
    )

    pred_rows = []

    for split_id, (tr, te) in enumerate(
        outer.split(X, y)
    ):

        repeat = split_id // 5
        fold = split_id % 5

        candidates = []

        for hidden in a.hidden:

            score = inner_score(
                X,
                y,
                tr,
                hidden,
                a.model,
                a.seed + split_id * 100,
                device
            )

            candidates.append(
                (score, hidden)
            )

            print(
                "split", split_id,
                "repeat", repeat,
                "fold", fold,
                "hidden", hidden,
                "inner_auc", score
            )

        candidates.sort(
            reverse=True,
            key=lambda z: z[0]
        )

        best_auc, hidden = candidates[0]

        # Outer-training-only scaling
        mean, std = training_node_scaler(
            X[tr]
        )

        Xs = apply_node_scaler(
            X,
            mean,
            std
        )

        gtr = build_graphs(
            Xs,
            y,
            tr
        )

        gte = build_graphs(
            Xs,
            y,
            te
        )

        model = train_fixed_epochs(
            gtr,
            a.model,
            hidden,
            epochs=120,
            seed=a.seed + split_id,
            device=device
        )

        auc, yt, st, idxs = evaluate(
            model,
            DataLoader(
                gte,
                batch_size=16
            ),
            device
        )

        ckpt = {
            "state_dict":
                model.state_dict(),

            "model_kind":
                a.model,

            "hidden":
                hidden,

            "ablation":
                "x_only_self_loops",

            "mean":
                mean.astype(np.float32),

            "std":
                std.astype(np.float32),

            "split":
                split_id,

            "repeat":
                repeat,

            "fold":
                fold,

            "test_indices":
                te,

            "inner_best_auc":
                best_auc,
        }

        torch.save(
            ckpt,
            out / f"split_{split_id}.pt"
        )

        for yy, ss, ii in zip(
            yt,
            st,
            idxs
        ):

            pred_rows.append({
                "split":
                    split_id,

                "repeat":
                    repeat,

                "fold":
                    fold,

                "subject_index":
                    int(ii),

                "subject_id":
                    ids[ii],

                "y":
                    int(yy),

                "score":
                    float(ss),

                "hidden":
                    hidden,

                "inner_best_auc":
                    best_auc,
            })

    pred = pd.DataFrame(
        pred_rows
    )

    pred.to_csv(
        out / "outer_predictions.csv",
        index=False
    )

    roc = roc_auc_score(
        pred.y,
        pred.score
    )

    pr = average_precision_score(
        pred.y,
        pred.score
    )

    metrics = {
        "roc_auc":
            float(roc),

        "pr_auc":
            float(pr),

        "device":
            str(device),

        "outer_cv":
            "5-fold x 3 repeats",

        "ablation":
            "x_only_self_loops",

        "uses_subject_specific_edges":
            False
    }

    (
        out / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2
        )
    )

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
