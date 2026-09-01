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
    p.add_argument("--msn", required=True)
    p.add_argument("--mind", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rf = pd.read_csv(a.rf)[
        ["subject_id", "y", "score"]
    ].rename(columns={"score": "rf"})

    msn = pd.read_csv(a.msn)[
        ["subject_id", "y", "score"]
    ].rename(columns={"score": "msn"})

    mind = pd.read_csv(a.mind)[
        ["subject_id", "y", "score"]
    ].rename(columns={"score": "mind"})

    df = (
        rf.merge(
            msn,
            on=["subject_id", "y"],
            validate="one_to_one",
        )
        .merge(
            mind,
            on=["subject_id", "y"],
            validate="one_to_one",
        )
    )

    assert len(df) == 71

    y = df.y.to_numpy(int)

    scores = {
        "ROI-RF": df.rf.to_numpy(float),
        "MSN-GAT": df.msn.to_numpy(float),
        "MIND-GAT": df.mind.to_numpy(float),
    }

    observed = {}

    for name, s in scores.items():

        pred = (s >= 0.5).astype(int)

        observed[name] = {
            "roc_auc":
                float(roc_auc_score(y, s)),

            "pr_auc":
                float(average_precision_score(y, s)),

            "balanced_accuracy":
                float(
                    balanced_accuracy_score(
                        y,
                        pred
                    )
                ),
        }

    pairs = [
        ("ROI-RF", "MSN-GAT"),
        ("ROI-RF", "MIND-GAT"),
        ("MSN-GAT", "MIND-GAT"),
    ]

    controls = np.where(y == 0)[0]
    cases = np.where(y == 1)[0]

    rng = np.random.default_rng(a.seed)

    result = {}

    for a_name, b_name in pairs:

        vals = {
            "roc_auc": [],
            "pr_auc": [],
            "balanced_accuracy": [],
        }

        sa = scores[a_name]
        sb = scores[b_name]

        for _ in range(a.bootstrap):

            c = rng.choice(
                controls,
                len(controls),
                replace=True,
            )

            z = rng.choice(
                cases,
                len(cases),
                replace=True,
            )

            idx = np.concatenate([c, z])

            yb = y[idx]
            xa = sa[idx]
            xb = sb[idx]

            vals["roc_auc"].append(
                roc_auc_score(yb, xa)
                - roc_auc_score(yb, xb)
            )

            vals["pr_auc"].append(
                average_precision_score(yb, xa)
                - average_precision_score(yb, xb)
            )

            vals["balanced_accuracy"].append(
                balanced_accuracy_score(
                    yb,
                    (xa >= 0.5).astype(int)
                )
                -
                balanced_accuracy_score(
                    yb,
                    (xb >= 0.5).astype(int)
                )
            )

        key = f"{a_name}_minus_{b_name}"

        result[key] = {}

        for metric, x in vals.items():

            x = np.asarray(x)

            result[key][metric] = {
                "observed":
                    float(
                        observed[a_name][metric]
                        - observed[b_name][metric]
                    ),

                "lower_95":
                    float(
                        np.percentile(x, 2.5)
                    ),

                "upper_95":
                    float(
                        np.percentile(x, 97.5)
                    ),

                "prob_delta_gt_0":
                    float(np.mean(x > 0)),
            }

    output = {
        "n": 71,
        "cases": int(y.sum()),
        "controls": int((y == 0).sum()),
        "observed": observed,
        "paired_bootstrap": result,
        "bootstrap_iterations": int(a.bootstrap),
    }

    (
        out / "three_model_comparison.json"
    ).write_text(
        json.dumps(output, indent=2)
    )

    print("\n======================================")
    print("THREE-MODEL EXTERNAL COMPARISON")
    print("======================================")

    for name, m in observed.items():

        print(
            f"{name:10s} "
            f"ROC={m['roc_auc']:.4f} "
            f"PR={m['pr_auc']:.4f} "
            f"BAcc={m['balanced_accuracy']:.4f}"
        )

    print("\nPAIRED DIFFERENCES")
    print("--------------------------------------")

    for pair, metrics in result.items():

        print("\n", pair)

        for metric, r in metrics.items():

            print(
                f"{metric:20s} "
                f"Δ={r['observed']:.4f} "
                f"95% CI "
                f"[{r['lower_95']:.4f}, "
                f"{r['upper_95']:.4f}]"
            )


if __name__ == "__main__":
    main()
