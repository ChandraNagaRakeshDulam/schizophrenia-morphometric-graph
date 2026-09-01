#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--rf", required=True)
    p.add_argument("--gat", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rf = pd.read_csv(a.rf)
    gat = pd.read_csv(a.gat)

    # Keep only required fields.
    rf = rf[
        ["subject_id", "y", "score"]
    ].rename(
        columns={"score": "rf_score"}
    )

    gat = gat[
        ["subject_id", "y", "score"]
    ].rename(
        columns={"score": "gat_score"}
    )

    df = rf.merge(
        gat,
        on=["subject_id", "y"],
        how="inner",
        validate="one_to_one",
    )

    assert len(df) == 71

    y = df["y"].to_numpy(dtype=int)

    rf_score = df["rf_score"].to_numpy(float)
    gat_score = df["gat_score"].to_numpy(float)

    # Fixed threshold used in both external evaluations.
    rf_pred = (rf_score >= 0.5).astype(int)
    gat_pred = (gat_score >= 0.5).astype(int)

    observed = {
        "rf_roc_auc":
            float(roc_auc_score(y, rf_score)),

        "gat_roc_auc":
            float(roc_auc_score(y, gat_score)),

        "delta_roc_rf_minus_gat":
            float(
                roc_auc_score(y, rf_score)
                - roc_auc_score(y, gat_score)
            ),

        "rf_pr_auc":
            float(
                average_precision_score(y, rf_score)
            ),

        "gat_pr_auc":
            float(
                average_precision_score(y, gat_score)
            ),

        "delta_pr_rf_minus_gat":
            float(
                average_precision_score(y, rf_score)
                - average_precision_score(y, gat_score)
            ),

        "rf_balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    rf_pred
                )
            ),

        "gat_balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    gat_pred
                )
            ),

        "delta_balanced_accuracy_rf_minus_gat":
            float(
                balanced_accuracy_score(y, rf_pred)
                - balanced_accuracy_score(y, gat_pred)
            ),
    }

    # -----------------------------------
    # Paired stratified bootstrap
    # -----------------------------------

    rng = np.random.default_rng(a.seed)

    controls = np.where(y == 0)[0]
    cases = np.where(y == 1)[0]

    delta_roc = []
    delta_pr = []
    delta_bacc = []

    for _ in range(a.bootstrap):

        c = rng.choice(
            controls,
            size=len(controls),
            replace=True,
        )

        s = rng.choice(
            cases,
            size=len(cases),
            replace=True,
        )

        idx = np.concatenate([c, s])

        yb = y[idx]

        rfb = rf_score[idx]
        gatb = gat_score[idx]

        delta_roc.append(
            roc_auc_score(yb, rfb)
            - roc_auc_score(yb, gatb)
        )

        delta_pr.append(
            average_precision_score(yb, rfb)
            - average_precision_score(yb, gatb)
        )

        delta_bacc.append(
            balanced_accuracy_score(
                yb,
                (rfb >= 0.5).astype(int)
            )
            -
            balanced_accuracy_score(
                yb,
                (gatb >= 0.5).astype(int)
            )
        )

    def summarize(x):
        x = np.asarray(x)

        return {
            "mean":
                float(np.mean(x)),

            "lower_95":
                float(
                    np.percentile(x, 2.5)
                ),

            "upper_95":
                float(
                    np.percentile(x, 97.5)
                ),

            "probability_delta_gt_0":
                float(np.mean(x > 0)),
        }

    bootstrap = {
        "delta_roc_rf_minus_gat":
            summarize(delta_roc),

        "delta_pr_rf_minus_gat":
            summarize(delta_pr),

        "delta_balanced_accuracy_rf_minus_gat":
            summarize(delta_bacc),
    }

    result = {
        "n": int(len(df)),
        "controls": int((y == 0).sum()),
        "cases": int(y.sum()),
        "comparison": "ROI-RF minus MSN-GAT",
        "paired_subject_level": True,
        "bootstrap_iterations": int(a.bootstrap),
        "observed": observed,
        "bootstrap": bootstrap,
    }

    (
        out / "rf_vs_msn_gat.json"
    ).write_text(
        json.dumps(result, indent=2)
    )

    df.to_csv(
        out / "paired_predictions.csv",
        index=False,
    )

    print(
        "\n======================================"
    )
    print("EXTERNAL ROI-RF vs MSN-GAT")
    print(
        "======================================"
    )

    print(
        f"ROI-RF ROC-AUC : "
        f"{observed['rf_roc_auc']:.4f}"
    )

    print(
        f"MSN-GAT ROC-AUC: "
        f"{observed['gat_roc_auc']:.4f}"
    )

    print(
        "\nΔ ROC-AUC (RF - GAT): "
        f"{observed['delta_roc_rf_minus_gat']:.4f}"
    )

    r = bootstrap[
        "delta_roc_rf_minus_gat"
    ]

    print(
        f"95% CI: "
        f"[{r['lower_95']:.4f}, "
        f"{r['upper_95']:.4f}]"
    )

    print(
        "Bootstrap P(Δ>0): "
        f"{r['probability_delta_gt_0']:.4f}"
    )

    print(
        "\nΔ PR-AUC (RF - GAT): "
        f"{observed['delta_pr_rf_minus_gat']:.4f}"
    )

    r = bootstrap[
        "delta_pr_rf_minus_gat"
    ]

    print(
        f"95% CI: "
        f"[{r['lower_95']:.4f}, "
        f"{r['upper_95']:.4f}]"
    )

    print(
        "\nΔ Balanced Accuracy "
        "(RF - GAT): "
        f"{observed['delta_balanced_accuracy_rf_minus_gat']:.4f}"
    )

    r = bootstrap[
        "delta_balanced_accuracy_rf_minus_gat"
    ]

    print(
        f"95% CI: "
        f"[{r['lower_95']:.4f}, "
        f"{r['upper_95']:.4f}]"
    )


if __name__ == "__main__":
    main()
