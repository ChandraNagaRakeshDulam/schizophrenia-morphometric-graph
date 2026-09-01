#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import RepeatedStratifiedKFold
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_utils import training_node_scaler, apply_node_scaler, make_graph
from models import GraphClassifier


def build_graphs(X, A, y, indices, density, rank_mode):
    return [
        make_graph(
            X[i], A[i], y[i],
            density, rank_mode, int(i)
        )
        for i in indices
    ]


def train_fixed_epochs(graphs, hidden, epochs, seed, device):

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

    ys = np.array([
        int(g.y.item()) for g in graphs
    ])

    n_pos = int(ys.sum())
    n_neg = int(len(ys) - n_pos)

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

    model.train()

    for _ in range(epochs):

        for batch in loader:

            batch = batch.to(device)

            opt.zero_grad()

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = loss_fn(
                logits,
                batch.y.view(-1)
            )

            loss.backward()
            opt.step()

    return model


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--density", type=float, default=0.20)
    p.add_argument("--hidden", type=int, default=64)

    p.add_argument("--rank-mode", default="absolute")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()

    d = np.load(
        a.dataset,
        allow_pickle=True
    )

    y = d["y"].astype(int)

    X = d["X_nodes"].astype(
        np.float32
    )

    A = d["A_msn"].astype(
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

    print("Device:", device)
    print("Subjects:", len(y))
    print("Fixed density:", a.density)
    print("Fixed hidden:", a.hidden)

    for split_id, (tr, te) in enumerate(
        outer.split(X, y)
    ):

        repeat = split_id // 5
        fold = split_id % 5

        # Training-fold-only scaling.
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
            A,
            y,
            tr,
            a.density,
            a.rank_mode
        )

        model = train_fixed_epochs(
            gtr,
            hidden=a.hidden,
            epochs=a.epochs,
            seed=a.seed + split_id,
            device=device
        )

        ckpt = {
            "state_dict":
                model.state_dict(),

            "model_kind":
                "gat",

            "representation":
                "msn",

            "hidden":
                int(a.hidden),

            "density":
                float(a.density),

            "rank_mode":
                a.rank_mode,

            "epochs":
                int(a.epochs),

            "mean":
                mean.astype(np.float32),

            "std":
                std.astype(np.float32),

            "split":
                int(split_id),

            "repeat":
                int(repeat),

            "fold":
                int(fold),

            "train_indices":
                np.asarray(tr, dtype=int),

            "test_indices":
                np.asarray(te, dtype=int),

            "uses_edge_weights":
                False,

            "analysis_role":
                (
                    "post-selection fixed-configuration "
                    "held-out explanation reference"
                ),
        }

        path = out / f"split_{split_id}.pt"

        torch.save(
            ckpt,
            path
        )

        print(
            f"saved split={split_id:02d} "
            f"repeat={repeat} fold={fold} "
            f"train={len(tr)} test={len(te)}"
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nCreated 15 fixed-config held-out checkpoints.")
    print("Output:", out)


if __name__ == "__main__":
    main()
