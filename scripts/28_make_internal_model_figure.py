#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


OUT = Path("results/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

TABLE_OUT = Path("results/paper")
TABLE_OUT.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Primary experiments only.
#
# Topology controls and smoke tests are intentionally omitted
# from the main comparison and will go in supplementary results.
# ==========================================================

EXPERIMENTS = [

    # ---------------- Traditional ML ----------------
    {
        "family": "Traditional ML",
        "representation": "ROI",
        "model": "LR",
        "label": "ROI-LR",
        "path":
            "results/ml/roi_lr_seed42.json/"
            "outer_predictions.csv",
    },
    {
        "family": "Traditional ML",
        "representation": "ROI",
        "model": "SVM",
        "label": "ROI-SVM",
        "path":
            "results/ml/roi_svm_seed42/"
            "outer_predictions.csv",
    },
    {
        "family": "Traditional ML",
        "representation": "ROI",
        "model": "RF",
        "label": "ROI-RF",
        "path":
            "results/ml/roi_rf_seed42/"
            "outer_predictions.csv",
    },

    {
        "family": "Traditional ML",
        "representation": "MSN",
        "model": "LR",
        "label": "MSN-LR",
        "path":
            "results/ml/msn_lr_seed42/"
            "outer_predictions.csv",
    },
    {
        "family": "Traditional ML",
        "representation": "MSN",
        "model": "SVM",
        "label": "MSN-SVM",
        "path":
            "results/ml/msn_svm_seed42/"
            "outer_predictions.csv",
    },
    {
        "family": "Traditional ML",
        "representation": "MSN",
        "model": "RF",
        "label": "MSN-RF",
        "path":
            "results/ml/msn_rf_seed42/"
            "outer_predictions.csv",
    },

    {
        "family": "Traditional ML",
        "representation": "MIND",
        "model": "LR",
        "label": "MIND-LR",
        "path":
            "results/ml/mind_lr_seed42/"
            "outer_predictions.csv",
    },
    {
        "family": "Traditional ML",
        "representation": "MIND",
        "model": "SVM",
        "label": "MIND-SVM",
        "path":
            "results/ml/mind_svm_seed42/"
            "outer_predictions.csv",
    },
    {
        "family": "Traditional ML",
        "representation": "MIND",
        "model": "RF",
        "label": "MIND-RF",
        "path":
            "results/ml/mind_rf_seed42/"
            "outer_predictions.csv",
    },

    # ---------------- MSN GNNs ----------------
    {
        "family": "GNN",
        "representation": "MSN",
        "model": "GCN",
        "label": "MSN-GCN",
        "path":
            "results/gnn/msn_gcn_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MSN",
        "model": "GraphSAGE",
        "label": "MSN-SAGE",
        "path":
            "results/gnn/msn_sage_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MSN",
        "model": "GIN",
        "label": "MSN-GIN",
        "path":
            "results/gnn/msn_gin_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MSN",
        "model": "GAT",
        "label": "MSN-GAT",
        "path":
            "results/gnn/msn_gat_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MSN",
        "model": "GATv2",
        "label": "MSN-GATv2",
        "path":
            "results/gnn/msn_gatv2_full/"
            "outer_predictions.csv",
    },

    # ---------------- MIND GNNs ----------------
    {
        "family": "GNN",
        "representation": "MIND",
        "model": "GCN",
        "label": "MIND-GCN",
        "path":
            "results/gnn/mind_gcn_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MIND",
        "model": "GraphSAGE",
        "label": "MIND-SAGE",
        "path":
            "results/gnn/mind_sage_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MIND",
        "model": "GIN",
        "label": "MIND-GIN",
        "path":
            "results/gnn/mind_gin_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MIND",
        "model": "GAT",
        "label": "MIND-GAT",
        "path":
            "results/gnn/mind_gat_full/"
            "outer_predictions.csv",
    },
    {
        "family": "GNN",
        "representation": "MIND",
        "model": "GATv2",
        "label": "MIND-GATv2",
        "path":
            "results/gnn/mind_gatv2_full/"
            "outer_predictions.csv",
    },
]


def repeat_auc(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "subject_id",
        "y",
        "score",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path}: missing columns {missing}"
        )

    # ML outputs do not contain an explicit repeat column.
    # Split IDs 0-4 = repeat 0,
    # 5-9 = repeat 1,
    # 10-14 = repeat 2.
    if "repeat" not in df.columns:
        df["repeat"] = (
            df["split"].astype(int) // 5
        )

    aucs = []

    n_subjects = []

    for repeat, q in df.groupby("repeat"):

        # Each subject should occur once per repeat,
        # because the five folds partition the dataset.
        if q["subject_id"].duplicated().any():
            raise RuntimeError(
                f"{path}: duplicate subjects "
                f"within repeat {repeat}"
            )

        auc = roc_auc_score(
            q["y"].to_numpy(),
            q["score"].to_numpy()
        )

        aucs.append(float(auc))
        n_subjects.append(
            int(q["subject_id"].nunique())
        )

    if len(aucs) != 3:
        raise RuntimeError(
            f"{path}: expected 3 repeats, "
            f"found {len(aucs)}"
        )

    if len(set(n_subjects)) != 1:
        raise RuntimeError(
            f"{path}: unequal repeat sizes "
            f"{n_subjects}"
        )

    return {
        "auc_repeat_0": aucs[0],
        "auc_repeat_1": aucs[1],
        "auc_repeat_2": aucs[2],
        "mean_roc_auc": float(
            np.mean(aucs)
        ),
        "sd_roc_auc": float(
            np.std(
                aucs,
                ddof=1
            )
        ),
        "n_per_repeat": n_subjects[0],
    }


rows = []

for exp in EXPERIMENTS:

    r = repeat_auc(
        exp["path"]
    )

    rows.append({
        **exp,
        **r,
    })


result = pd.DataFrame(rows)

result.to_csv(
    TABLE_OUT /
    "table_internal_model_comparison.csv",
    index=False
)


# ==========================================================
# Print numerical table
# ==========================================================

print()
print(
    "======================================"
)
print(
    "INTERNAL MODEL COMPARISON"
)
print(
    "======================================"
)

for family in [
    "Traditional ML",
    "GNN"
]:

    print()
    print(family)
    print("-" * 50)

    q = result[
        result.family == family
    ]

    for _, r in q.iterrows():

        print(
            f"{r['label']:12s} "
            f"{r['mean_roc_auc']:.4f} "
            f"± {r['sd_roc_auc']:.4f}"
        )


# ==========================================================
# FIGURE 2
# Two-panel internal comparison
# ==========================================================

ml = result[
    result.family == "Traditional ML"
].copy()

gnn = result[
    result.family == "GNN"
].copy()


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 7),
    sharex=True
)


def plot_panel(ax, df, title):

    # Reverse so first entry appears at top.
    q = df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(q))

    ax.errorbar(
        q["mean_roc_auc"],
        y,
        xerr=q["sd_roc_auc"],
        fmt="o",
        capsize=4,
        markersize=7,
        linewidth=1.4,
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        q["label"]
    )

    ax.set_xlim(
        0.50,
        0.75
    )

    ax.set_xlabel(
        "ROC-AUC (mean ± SD across 3 repeats)"
    )

    ax.set_title(title)

    ax.grid(
        axis="x",
        alpha=0.2
    )

    for yy, (_, r) in zip(
        y,
        q.iterrows()
    ):

        ax.text(
            r["mean_roc_auc"] + 0.007,
            yy,
            f"{r['mean_roc_auc']:.3f}",
            va="center",
            fontsize=8,
        )


plot_panel(
    axes[0],
    ml,
    "A. Conventional machine learning"
)

plot_panel(
    axes[1],
    gnn,
    "B. Graph neural networks"
)

fig.suptitle(
    "Internal schizophrenia classification performance",
    fontsize=14
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

for ext in [
    "png",
    "svg",
    "pdf",
]:

    path = (
        OUT /
        f"figure2_internal_model_comparison.{ext}"
    )

    fig.savefig(
        path,
        dpi=300 if ext == "png" else None,
        bbox_inches="tight"
    )

    print("Created:", path)

plt.close(fig)


# ==========================================================
# Also save ranking table
# ==========================================================

ranking = (
    result[
        [
            "family",
            "representation",
            "model",
            "label",
            "mean_roc_auc",
            "sd_roc_auc",
            "auc_repeat_0",
            "auc_repeat_1",
            "auc_repeat_2",
        ]
    ]
    .sort_values(
        "mean_roc_auc",
        ascending=False
    )
)

ranking.to_csv(
    TABLE_OUT /
    "table_internal_model_ranking.csv",
    index=False
)

print()
print(
    "======================================"
)
print(
    "TOP INTERNAL MODELS"
)
print(
    "======================================"
)

print(
    ranking.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
