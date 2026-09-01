#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from torch_geometric.loader import DataLoader

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from graph_utils import (
    apply_node_scaler,
    make_graph,
)

from models import GraphClassifier


def build_graphs(
    X,
    A,
    y,
    density,
    rank_mode
):
    return [
        make_graph(
            X[i],
            A[i],
            y[i],
            density,
            rank_mode,
            int(i),
        )
        for i in range(len(y))
    ]


@torch.no_grad()
def predict(model, graphs, device):

    model.eval()

    loader = DataLoader(
        graphs,
        batch_size=16,
        shuffle=False,
    )

    ys = []
    scores = []
    idxs = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        probs = torch.sigmoid(logits)

        ys.extend(
            batch.y.view(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        scores.extend(
            probs.cpu()
            .numpy()
            .tolist()
        )

        idxs.extend(
            batch.sample_idx.view(-1)
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        np.asarray(ys, dtype=int),
        np.asarray(scores, dtype=float),
        np.asarray(idxs, dtype=int),
    )


def metric_dict(y, score, threshold=0.5):

    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "roc_auc":
            float(roc_auc_score(y, score)),

        "pr_auc":
            float(
                average_precision_score(
                    y,
                    score
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred
                )
            ),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "precision":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0
                )
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stratified_bootstrap(
    y,
    score,
    threshold=0.5,
    n_boot=10000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]

    names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
    ]

    vals = {
        k: []
        for k in names
    }

    for _ in range(n_boot):

        bneg = rng.choice(
            neg,
            size=len(neg),
            replace=True,
        )

        bpos = rng.choice(
            pos,
            size=len(pos),
            replace=True,
        )

        idx = np.concatenate(
            [bneg, bpos]
        )

        m = metric_dict(
            y[idx],
            score[idx],
            threshold,
        )

        for k in names:
            vals[k].append(m[k])

    cis = {}

    for k in names:

        x = np.asarray(
            vals[k],
            dtype=float,
        )

        cis[k] = {
            "lower_95":
                float(
                    np.nanpercentile(
                        x,
                        2.5
                    )
                ),

            "upper_95":
                float(
                    np.nanpercentile(
                        x,
                        97.5
                    )
                ),
        }

    return cis


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        required=True,
    )

    p.add_argument(
        "--external",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    a = p.parse_args()

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

    # ---------------------------------------
    # Load FROZEN internal model
    # ---------------------------------------

    ckpt = torch.load(
        a.checkpoint,
        map_location=device,
        weights_only=False,
    )

    print(
        "Frozen model:",
        a.checkpoint
    )

    print(
        "Model:",
        ckpt["model_kind"]
    )

    print(
        "Representation:",
        ckpt["representation"]
    )

    print(
        "Density:",
        ckpt["density"]
    )

    print(
        "Hidden:",
        ckpt["hidden"]
    )

    print(
        "Rank mode:",
        ckpt["rank_mode"]
    )

    print(
        "Epochs:",
        ckpt["epochs"]
    )

    print(
        "Edge weights used:",
        ckpt["uses_edge_weights"]
    )

    # These are the expected frozen choices.
    assert ckpt["model_kind"] == "gat"
    assert ckpt["representation"] == "msn"
    assert float(ckpt["density"]) == 0.20
    assert int(ckpt["hidden"]) == 64
    assert ckpt["rank_mode"] == "absolute"
    assert int(ckpt["epochs"]) == 120
    assert ckpt["uses_edge_weights"] is False

    # ---------------------------------------
    # Load external dataset
    # ---------------------------------------

    d = np.load(
        a.external,
        allow_pickle=True,
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(
        np.float32
    )

    A = d["A_msn"].astype(
        np.float32
    )

    roi_names = d[
        "roi_names"
    ].astype(str)

    feature_names = d[
        "feature_names"
    ].astype(str)

    print("\nExternal dataset")
    print("----------------------------")

    print(
        "Subjects:",
        len(y)
    )

    print(
        "Cases:",
        int(y.sum())
    )

    print(
        "Controls:",
        int((y == 0).sum())
    )

    print(
        "X:",
        X.shape
    )

    print(
        "A:",
        A.shape
    )

    assert len(y) == 71
    assert int(y.sum()) == 46
    assert int((y == 0).sum()) == 25

    assert X.shape == (
        71,
        68,
        5
    )

    assert A.shape == (
        71,
        68,
        68
    )

    # ---------------------------------------
    # Cross-dataset representation check
    # ---------------------------------------

    assert np.array_equal(
        roi_names,
        np.asarray(
            ckpt["roi_names"]
        ).astype(str)
    )

    assert np.array_equal(
        feature_names,
        np.asarray(
            ckpt["feature_names"]
        ).astype(str)
    )

    # ---------------------------------------
    # IMPORTANT:
    # Apply INTERNAL scaler to external data.
    #
    # No fitting occurs on ds004302.
    # ---------------------------------------

    mean = np.asarray(
        ckpt["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        ckpt["std"],
        dtype=np.float32,
    )

    Xs = apply_node_scaler(
        X,
        mean,
        std,
    )

    # ---------------------------------------
    # Construct external graphs using
    # FROZEN density/rank mode.
    # ---------------------------------------

    graphs = build_graphs(
        Xs,
        A,
        y,
        float(ckpt["density"]),
        ckpt["rank_mode"],
    )

    # ---------------------------------------
    # Restore frozen model
    # ---------------------------------------

    model = GraphClassifier(
        X.shape[-1],
        int(ckpt["hidden"]),
        ckpt["model_kind"],
    ).to(device)

    model.load_state_dict(
        ckpt["state_dict"]
    )

    # ---------------------------------------
    # ONE external prediction pass
    # ---------------------------------------

    yt, score, idx = predict(
        model,
        graphs,
        device,
    )

    assert np.array_equal(
        yt,
        y[idx]
    )

    pred = (
        score >= a.threshold
    ).astype(int)

    pred_df = pd.DataFrame({
        "subject_index":
            idx,

        "subject_id":
            ids[idx],

        "y":
            yt,

        "score":
            score,

        "pred":
            pred,
    })

    pred_df.to_csv(
        out /
        "external_predictions.csv",
        index=False,
    )

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    metrics = metric_dict(
        yt,
        score,
        a.threshold,
    )

    cis = stratified_bootstrap(
        yt,
        score,
        threshold=a.threshold,
        n_boot=a.bootstrap,
        seed=a.seed,
    )

    result = {
        "dataset":
            str(a.external),

        "checkpoint":
            str(a.checkpoint),

        "n":
            int(len(yt)),

        "cases":
            int(yt.sum()),

        "controls":
            int((yt == 0).sum()),

        "external_prevalence":
            float(yt.mean()),

        "pr_auc_random_baseline":
            float(yt.mean()),

        "threshold":
            float(a.threshold),

        "threshold_tuned_on_external":
            False,

        "external_data_used_for_model_selection":
            False,

        "density":
            float(ckpt["density"]),

        "hidden":
            int(ckpt["hidden"]),

        "rank_mode":
            ckpt["rank_mode"],

        "uses_edge_weights":
            bool(
                ckpt[
                    "uses_edge_weights"
                ]
            ),

        "metrics":
            metrics,

        "bootstrap_95_ci":
            cis,

        "bootstrap_iterations":
            int(a.bootstrap),

        "bootstrap_method":
            (
                "subject-level stratified "
                "percentile bootstrap"
            ),
    }

    (
        out /
        "external_metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    # ---------------------------------------
    # Display important results
    # ---------------------------------------

    print(
        "\n======================================"
    )

    print(
        "FROZEN EXTERNAL VALIDATION"
    )

    print(
        "======================================"
    )

    for k in [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    ]:

        lo = cis[k]["lower_95"]
        hi = cis[k]["upper_95"]

        print(
            f"{k:20s} "
            f"{metrics[k]:.4f} "
            f"[{lo:.4f}, {hi:.4f}]"
        )

    print(
        "\nConfusion matrix:"
    )

    print(
        "TN:",
        metrics["tn"],
        "FP:",
        metrics["fp"],
        "FN:",
        metrics["fn"],
        "TP:",
        metrics["tp"],
    )

    print(
        "\nExternal SZ prevalence / "
        "random PR-AUC baseline:",
        f"{yt.mean():.4f}",
    )

    print(
        "\nPredictions:",
        out /
        "external_predictions.csv"
    )

    print(
        "Metrics:",
        out /
        "external_metrics.json"
    )


if __name__ == "__main__":
    main()
