#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path("results/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, body="", fontsize=10):

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
    )

    if body:
        ax.text(
            x + w / 2,
            y + h * 0.35,
            body,
            ha="center",
            va="center",
            fontsize=fontsize - 1,
            linespacing=1.3,
        )


def arrow(ax, x1, y1, x2, y2):

    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
    )

    ax.add_patch(a)


fig, ax = plt.subplots(
    figsize=(17, 10)
)

ax.set_xlim(0, 17)
ax.set_ylim(0, 10)
ax.axis("off")


# ==========================================================
# DEVELOPMENT COHORT
# ==========================================================

ax.text(
    0.5,
    9.55,
    "Development cohort: ds000030",
    fontsize=14,
    fontweight="bold",
)

ax.text(
    0.5,
    9.15,
    "N = 175 | 125 controls | 50 schizophrenia",
    fontsize=10,
)


# T1 MRI
box(
    ax,
    0.5, 7.4,
    2.0, 1.2,
    "T1-weighted MRI",
    "Structural MRI"
)

# FreeSurfer
box(
    ax,
    3.1, 7.4,
    2.3, 1.2,
    "FreeSurfer 7.4.1",
    "recon-all\nmanual surface QC"
)

# DK68
box(
    ax,
    6.0, 7.4,
    2.3, 1.2,
    "DK68 parcellation",
    "34 left + 34 right\ncortical ROIs"
)

# Features
box(
    ax,
    8.9, 7.4,
    2.5, 1.2,
    "Morphometric features",
    "CT | MC | Vol | SD | SA"
)

arrow(ax, 2.5, 8.0, 3.1, 8.0)
arrow(ax, 5.4, 8.0, 6.0, 8.0)
arrow(ax, 8.3, 8.0, 8.9, 8.0)


# ==========================================================
# REPRESENTATIONS
# ==========================================================

ax.text(
    0.5,
    6.5,
    "Subject representations",
    fontsize=13,
    fontweight="bold",
)

box(
    ax,
    1.0, 5.0,
    3.0, 1.2,
    "Regional morphology",
    "68 × 5 = 340 features"
)

box(
    ax,
    5.0, 5.0,
    3.0, 1.2,
    "MSN",
    "Subject-specific\nmorphometric similarity"
)

box(
    ax,
    9.0, 5.0,
    3.0, 1.2,
    "MIND",
    "Vertex-distribution-based\nmorphometric similarity"
)

# Feature box branches to representations.
arrow(ax, 10.15, 7.4, 2.5, 6.2)
arrow(ax, 10.15, 7.4, 6.5, 6.2)
arrow(ax, 10.15, 7.4, 10.5, 6.2)


# ==========================================================
# MODELS
# ==========================================================

ax.text(
    0.5,
    4.25,
    "Predictive modeling",
    fontsize=13,
    fontweight="bold",
)

box(
    ax,
    1.0, 2.75,
    3.0, 1.2,
    "Conventional ML",
    "Logistic regression\nSVM | Random forest"
)

box(
    ax,
    5.0, 2.75,
    3.0, 1.2,
    "Graph neural networks",
    "GCN | GraphSAGE | GIN\nGAT | GATv2"
)

box(
    ax,
    9.0, 2.75,
    3.0, 1.2,
    "Explainability",
    "GNNExplainer\nROI | feature | edge attribution"
)

box(
    ax,
    13.0, 2.75,
    3.2, 1.2,
    "Reliability analysis",
    "Stability | faithfulness\ncross-cohort reproducibility"
)

arrow(ax, 2.5, 5.0, 2.5, 3.95)
arrow(ax, 6.5, 5.0, 6.5, 3.95)
arrow(ax, 10.5, 5.0, 6.9, 3.95)

arrow(ax, 8.0, 3.35, 9.0, 3.35)
arrow(ax, 12.0, 3.35, 13.0, 3.35)


# ==========================================================
# EXTERNAL VALIDATION
# ==========================================================

ax.text(
    0.5,
    1.85,
    "Independent external validation",
    fontsize=13,
    fontweight="bold",
)

box(
    ax,
    1.0, 0.35,
    3.5, 1.15,
    "ds004302",
    "N = 71\n25 controls | 46 schizophrenia"
)

box(
    ax,
    5.1, 0.35,
    3.5, 1.15,
    "Frozen prediction",
    "ROI-RF | MSN-GAT | MIND-GAT\nNo external tuning"
)

box(
    ax,
    9.2, 0.35,
    3.5, 1.15,
    "External explanations",
    "Frozen MSN-GAT\nROI + feature attribution"
)

box(
    ax,
    13.3, 0.35,
    3.2, 1.15,
    "Generalization",
    "Classification\nreproducibility + faithfulness"
)

arrow(ax, 4.5, 0.92, 5.1, 0.92)
arrow(ax, 8.6, 0.92, 9.2, 0.92)
arrow(ax, 12.7, 0.92, 13.3, 0.92)


# Link development/frozen training to external prediction.
arrow(
    ax,
    7.0, 2.75,
    6.9, 1.50
)

# Link reliability/explanation framework to external analysis.
arrow(
    ax,
    14.6, 2.75,
    14.7, 1.50
)


fig.suptitle(
    "Explainable subject-specific morphometric graph learning "
    "for schizophrenia classification",
    fontsize=17,
    fontweight="bold",
    y=0.98,
)


for ext in ["png", "svg", "pdf"]:

    path = OUT / f"figure1_study_workflow.{ext}"

    fig.savefig(
        path,
        dpi=300 if ext == "png" else None,
        bbox_inches="tight",
    )

    print("Created:", path)


plt.close(fig)

print()
print("Figure 1 complete.")
