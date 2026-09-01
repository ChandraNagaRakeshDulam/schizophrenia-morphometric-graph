#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_utils import (
    training_node_scaler,
    apply_node_scaler,
    make_graph,
)
from models import GraphClassifier


def build_graphs(X, A, y, indices, density, rank_mode):
    return [
        make_graph(
            X[i],
            A[i],
            y[i],
            density,
            rank_mode,
            int(i),
        )
        for i in indices
    ]


def train_fixed_epochs(
    graphs,
    kind,
    hidden,
    epochs,
    seed,
    device,
):
    torch.manual_seed(seed)

    model = GraphClassifier(
        graphs[0].num_node_features,
        hidden,
        kind,
    ).to(device)

    loader = DataLoader(
        graphs,
        batch_size=min(16, len(graphs)),
        shuffle=True,
    )

    ys = np.array(
        [int(g.y.item()) for g in graphs]
    )

    n_pos = int(ys.sum())
    n_neg = int(len(ys) - n_pos)

    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        device=device,
        dtype=torch.float32,
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    model.train()

    for _ in range(int(epochs)):
        for batch in loader:
            batch = batch.to(device)

            opt.zero_grad()

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            loss = loss_fn(
                logits,
                batch.y.view(-1),
            )

            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def predict_scores(model, graphs, device):
    model.eval()

    loader = DataLoader(
        graphs,
        batch_size=16,
        shuffle=False,
    )

    ys = []
    scores = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        prob = torch.sigmoid(logits)

        ys.extend(
            batch.y.view(-1).cpu().numpy().tolist()
        )

        scores.extend(
            prob.cpu().numpy().tolist()
        )

    return (
        np.asarray(ys, dtype=int),
        np.asarray(scores, dtype=float),
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30],
    )

    p.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[32, 64],
    )

    p.add_argument(
        "--rank-mode",
        choices=["absolute", "positive"],
        default="absolute",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    a = p.parse_args()

    # -------------------------------------------------
    # Load INTERNAL development dataset only.
    # -------------------------------------------------

    d = np.load(
        a.dataset,
        allow_pickle=True,
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(np.float32)
    A = d["A_msn"].astype(np.float32)

    roi_names = d["roi_names"].astype(str)
    feature_names = d["feature_names"].astype(str)

    assert X.shape[1:] == (68, 5)
    assert A.shape[1:] == (68, 68)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Subjects:", len(y))
    print("Cases:", int(y.sum()))
    print("Controls:", int((y == 0).sum()))

    # -------------------------------------------------
    # Final hyperparameter selection.
    #
    # IMPORTANT:
    # This is development-only CV.
    # External ds004302 is not loaded anywhere.
    # -------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=a.seed,
    )

    rows = []

    configs = [
        (density, hidden)
        for density in a.densities
        for hidden in a.hidden
    ]

    print("\nCandidate configurations:")
    for c in configs:
        print(c)

    for split_id, (tr, va) in enumerate(
        cv.split(np.zeros(len(y)), y)
    ):

        repeat = split_id // 5
        fold = split_id % 5

        # Training-fold-only feature scaling.
        mean, std = training_node_scaler(
            X[tr]
        )

        Xs = apply_node_scaler(
            X,
            mean,
            std,
        )

        for density, hidden in configs:

            print(
                f"split={split_id:02d} "
                f"repeat={repeat} "
                f"fold={fold} "
                f"density={density:.2f} "
                f"hidden={hidden}"
            )

            gtr = build_graphs(
                Xs,
                A,
                y,
                tr,
                density,
                a.rank_mode,
            )

            gva = build_graphs(
                Xs,
                A,
                y,
                va,
                density,
                a.rank_mode,
            )

            # Keep training recipe identical to
            # scripts/10_train_gnn.py.
            model = train_fixed_epochs(
                gtr,
                kind="gat",
                hidden=hidden,
                epochs=a.epochs,
                seed=a.seed + split_id,
                device=device,
            )

            yt, score = predict_scores(
                model,
                gva,
                device,
            )

            auc = roc_auc_score(
                yt,
                score,
            )

            print(
                f"  validation ROC-AUC = "
                f"{auc:.6f}"
            )

            rows.append({
                "split": split_id,
                "repeat": repeat,
                "fold": fold,
                "density": density,
                "hidden": hidden,
                "roc_auc": float(auc),
            })

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    cv_df = pd.DataFrame(rows)

    cv_df.to_csv(
        out / "cv_scores.csv",
        index=False,
    )

    # -------------------------------------------------
    # Aggregate the 15 validation AUC values for
    # each candidate.
    # -------------------------------------------------

    summary = (
        cv_df
        .groupby(
            ["density", "hidden"],
            as_index=False,
        )
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            median_roc_auc=("roc_auc", "median"),
            n=("roc_auc", "size"),
        )
    )

    # Primary rule:
    # highest mean CV ROC-AUC.
    #
    # Deterministic tie-breaking decided BEFORE
    # external evaluation:
    # 1. lower density
    # 2. lower hidden dimension
    summary = summary.sort_values(
        by=[
            "mean_roc_auc",
            "density",
            "hidden",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    summary.to_csv(
        out / "cv_summary.csv",
        index=False,
    )

    print("\n==============================")
    print("DEVELOPMENT-ONLY CV SUMMARY")
    print("==============================")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    best = summary.iloc[0]

    best_density = float(
        best["density"]
    )

    best_hidden = int(
        best["hidden"]
    )

    print("\nSELECTED CONFIGURATION")
    print("------------------------------")
    print("density:", best_density)
    print("hidden :", best_hidden)
    print(
        "mean development CV AUC:",
        float(best["mean_roc_auc"]),
    )

    # -------------------------------------------------
    # Fit final scaler on ALL 175 internal subjects.
    # -------------------------------------------------

    mean, std = training_node_scaler(X)

    Xs = apply_node_scaler(
        X,
        mean,
        std,
    )

    all_idx = np.arange(
        len(y),
        dtype=int,
    )

    graphs = build_graphs(
        Xs,
        A,
        y,
        all_idx,
        best_density,
        a.rank_mode,
    )

    # -------------------------------------------------
    # Train final frozen model on ALL development data.
    # -------------------------------------------------

    final_model = train_fixed_epochs(
        graphs,
        kind="gat",
        hidden=best_hidden,
        epochs=a.epochs,
        seed=a.seed,
        device=device,
    )

    checkpoint = {
        "state_dict":
            final_model.state_dict(),

        "model_kind":
            "gat",

        "representation":
            "msn",

        "hidden":
            best_hidden,

        "density":
            best_density,

        "rank_mode":
            a.rank_mode,

        "epochs":
            int(a.epochs),

        "learning_rate":
            1e-3,

        "weight_decay":
            1e-4,

        "batch_size":
            16,

        "uses_edge_weights":
            False,

        "seed":
            int(a.seed),

        "mean":
            mean.astype(np.float32),

        "std":
            std.astype(np.float32),

        "roi_names":
            roi_names,

        "feature_names":
            feature_names,

        "development_dataset":
            str(a.dataset),

        "n_train":
            int(len(y)),

        "n_cases":
            int(y.sum()),

        "n_controls":
            int((y == 0).sum()),

        "selection_rule":
            (
                "highest mean ROC-AUC across "
                "5-fold x 3-repeat development-only CV; "
                "ties resolved by lower density then "
                "lower hidden dimension"
            ),

        "selected_cv_mean_auc":
            float(best["mean_roc_auc"]),

        "selected_cv_sd_auc":
            float(best["sd_roc_auc"]),
    }

    final_path = (
        out /
        "final_msn_gat_ds000030.pt"
    )

    torch.save(
        checkpoint,
        final_path,
    )

    selection = {
        "development_dataset":
            str(a.dataset),

        "external_dataset_used_for_selection":
            False,

        "representation":
            "MSN",

        "model":
            "GAT",

        "selected_density":
            best_density,

        "selected_hidden":
            best_hidden,

        "rank_mode":
            a.rank_mode,

        "epochs":
            int(a.epochs),

        "optimizer":
            "AdamW",

        "learning_rate":
            1e-3,

        "weight_decay":
            1e-4,

        "uses_edge_weights":
            False,

        "training_subjects":
            int(len(y)),

        "cases":
            int(y.sum()),

        "controls":
            int((y == 0).sum()),

        "selection_cv":
            "RepeatedStratifiedKFold 5x3",

        "selection_metric":
            "ROC-AUC",

        "selection_rule":
            (
                "Maximum mean development-only CV ROC-AUC; "
                "tie-break lower density then lower hidden"
            ),
    }

    (
        out /
        "selection.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        )
    )

    print("\n==============================")
    print("FINAL MODEL FROZEN")
    print("==============================")

    print(
        "Checkpoint:",
        final_path,
    )

    print(
        "Selection record:",
        out / "selection.json",
    )

    print(
        "\nIMPORTANT: Do not modify model settings "
        "after viewing ds004302 outcomes."
    )


if __name__ == "__main__":
    main()
