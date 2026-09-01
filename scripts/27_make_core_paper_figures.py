#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path("results")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("Created:", OUT / f"{name}.png")
    print("Created:", OUT / f"{name}.svg")


# ==========================================================
# FIGURE 3
# External classification ROC-AUC with bootstrap 95% CI
# ==========================================================

ext = pd.read_csv(
    ROOT / "paper" / "table_external_classification.csv"
)

models = ext["Model"].tolist()
roc = ext["ROC-AUC"].to_numpy(float)
lo = ext["ROC-AUC CI lower"].to_numpy(float)
hi = ext["ROC-AUC CI upper"].to_numpy(float)

yerr = np.vstack([
    roc - lo,
    hi - roc,
])

fig, ax = plt.subplots(figsize=(7, 4.8))

x = np.arange(len(models))

ax.errorbar(
    x,
    roc,
    yerr=yerr,
    fmt="o",
    capsize=6,
    markersize=8,
    linewidth=1.8,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel("ROC-AUC")
ax.set_title("Independent external classification performance")

ax.set_ylim(0.25, 0.90)

for i, value in enumerate(roc):
    ax.text(
        i,
        value + 0.035,
        f"{value:.3f}",
        ha="center",
        fontsize=10,
    )

save(fig, "figure3_external_classification")


# ==========================================================
# FIGURE 4
# Internal vs external schizophrenia ROI importance
# ==========================================================

roi = pd.read_csv(
    ROOT
    / "explain_cross_cohort"
    / "msn_gat"
    / "roi_cross_cohort_comparison.csv"
)

sz = roi[roi["group"] == "SZ"].copy()

rho = spearmanr(
    sz["importance_internal"],
    sz["importance_external"]
).statistic

fig, ax = plt.subplots(figsize=(6.5, 6))

ax.scatter(
    sz["importance_internal"],
    sz["importance_external"],
    s=34,
    alpha=0.8,
)

ax.set_xlabel("Internal mean ROI importance")
ax.set_ylabel("External mean ROI importance")

ax.set_title(
    f"Cross-cohort schizophrenia ROI importance\n"
    f"Spearman ρ = {rho:.3f}"
)

# Label ROIs with largest combined importance.
sz["combined"] = (
    sz["importance_internal"]
    + sz["importance_external"]
)

top = sz.nlargest(10, "combined")

for _, r in top.iterrows():
    ax.annotate(
        r["roi"],
        (
            r["importance_internal"],
            r["importance_external"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )

save(fig, "figure4_cross_cohort_roi_reproducibility")


# ==========================================================
# FIGURE 5
# Morphometric feature reproducibility
# ==========================================================

feat = pd.read_csv(
    ROOT
    / "paper"
    / "table_feature_reproducibility.csv"
)

szf = feat[
    feat["Group"] == "SZ"
].copy()

order = ["CT", "MC", "SD", "SA", "Vol"]

szf["Feature"] = pd.Categorical(
    szf["Feature"],
    categories=order,
    ordered=True
)

szf = szf.sort_values("Feature")

x = np.arange(len(szf))
width = 0.36

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    x - width / 2,
    szf["Internal importance"],
    width,
    label="Internal",
)

ax.bar(
    x + width / 2,
    szf["External importance"],
    width,
    label="External",
)

ax.set_xticks(x)
ax.set_xticklabels(szf["Feature"])

ax.set_ylabel("Normalized feature importance")
ax.set_title(
    "Schizophrenia morphometric feature importance\n"
    "Internal vs independent external cohort"
)

ax.legend(frameon=False)

rho_feat = float(
    szf["Feature Spearman"].iloc[0]
)

ax.text(
    0.98,
    0.95,
    f"Spearman ρ = {rho_feat:.2f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
)

save(fig, "figure5_feature_reproducibility")


# ==========================================================
# FIGURE 6
# External multi-k explanation faithfulness
# ==========================================================

faith = pd.read_csv(
    ROOT
    / "paper"
    / "table_external_faithfulness.csv"
)

all_subjects = faith[
    faith["group"] == "ALL"
].sort_values("k")

sz_subjects = faith[
    faith["group"] == "SZ"
].sort_values("k")

fig, ax = plt.subplots(figsize=(8, 5.2))

ax.errorbar(
    all_subjects["k"],
    all_subjects["advantage"],
    yerr=np.vstack([
        all_subjects["advantage"]
        - all_subjects["ci_lower"],
        all_subjects["ci_upper"]
        - all_subjects["advantage"],
    ]),
    marker="o",
    capsize=5,
    linewidth=1.8,
    label="All external subjects",
)

ax.errorbar(
    sz_subjects["k"],
    sz_subjects["advantage"],
    yerr=np.vstack([
        sz_subjects["advantage"]
        - sz_subjects["ci_lower"],
        sz_subjects["ci_upper"]
        - sz_subjects["advantage"],
    ]),
    marker="s",
    capsize=5,
    linewidth=1.8,
    label="Schizophrenia",
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks([1, 5, 10, 15, 20])

ax.set_xlabel("Number of perturbed top-ranked ROIs (k)")
ax.set_ylabel(
    "Faithfulness advantage\n"
    "(top-k confidence drop − random-k drop)"
)

ax.set_title(
    "External explanation faithfulness"
)

ax.legend(frameon=False)

save(fig, "figure6_external_faithfulness")


print()
print("======================================")
print("CORE PAPER FIGURES COMPLETE")
print("======================================")

for p in sorted(OUT.glob("figure*")):
    print(p)
