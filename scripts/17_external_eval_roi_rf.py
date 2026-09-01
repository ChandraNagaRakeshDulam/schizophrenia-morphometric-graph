#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(y, score, threshold=0.5):

    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "sensitivity": float(
            tp / (tp + fn) if tp + fn else np.nan
        ),
        "specificity": float(
            tn / (tn + fp) if tn + fp else np.nan
        ),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(
    y,
    score,
    threshold=0.5,
    n_boot=10000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    controls = np.where(y == 0)[0]
    cases = np.where(y == 1)[0]

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

    values = {k: [] for k in names}

    for _ in range(n_boot):

        c = rng.choice(
            controls,
            len(controls),
            replace=True,
        )

        s = rng.choice(
            cases,
            len(cases),
            replace=True,
        )

        idx = np.concatenate([c, s])

        m = calculate_metrics(
            y[idx],
            score[idx],
            threshold,
        )

        for k in names:
            values[k].append(m[k])

    return {
        k: {
            "lower_95": float(
                np.nanpercentile(values[k], 2.5)
            ),
            "upper_95": float(
                np.nanpercentile(values[k], 97.5)
            ),
        }
        for k in names
    }


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--external", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Frozen development-only RF
    # -------------------------------
    frozen = joblib.load(a.model)

    assert frozen["model_kind"] == "rf"
    assert frozen["representation"] == "roi"
    assert frozen["threshold"] == 0.5
    assert frozen["external_data_used_for_selection"] is False

    model = frozen["model"]

    print("Frozen ROI-RF")
    print("------------------------------")
    print("Parameters:", frozen["best_params"])
    print(
        "Development selection AUC:",
        frozen["selected_cv_mean_auc"]
    )

    # -------------------------------
    # External dataset
    # -------------------------------
    d = np.load(
        a.external,
        allow_pickle=True,
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X_nodes = d["X_nodes"].astype(np.float32)

    assert X_nodes.shape == (71, 68, 5)

    assert np.array_equal(
        d["roi_names"].astype(str),
        np.asarray(frozen["roi_names"]).astype(str),
    )

    assert np.array_equal(
        d["feature_names"].astype(str),
        np.asarray(frozen["feature_names"]).astype(str),
    )

    # Identical flattening used internally.
    X = X_nodes.reshape(len(y), -1)

    assert X.shape == (71, 340)

    print("\nExternal subjects:", len(y))
    print("Cases:", int(y.sum()))
    print("Controls:", int((y == 0).sum()))

    # IMPORTANT:
    # No fit / tuning occurs here.
    score = model.predict_proba(X)[:, 1]

    threshold = 0.5
    pred = (score >= threshold).astype(int)

    predictions = pd.DataFrame({
        "subject_id": ids,
        "y": y,
        "score": score,
        "pred": pred,
    })

    predictions.to_csv(
        out / "external_predictions.csv",
        index=False,
    )

    metrics = calculate_metrics(
        y,
        score,
        threshold,
    )

    cis = bootstrap_ci(
        y,
        score,
        threshold=threshold,
        n_boot=a.bootstrap,
        seed=a.seed,
    )

    result = {
        "dataset": str(a.external),
        "model": str(a.model),
        "n": int(len(y)),
        "cases": int(y.sum()),
        "controls": int((y == 0).sum()),
        "external_prevalence": float(y.mean()),
        "pr_auc_random_baseline": float(y.mean()),
        "threshold": threshold,
        "threshold_tuned_on_external": False,
        "external_data_used_for_model_selection": False,
        "best_params": frozen["best_params"],
        "metrics": metrics,
        "bootstrap_95_ci": cis,
        "bootstrap_iterations": int(a.bootstrap),
        "bootstrap_method":
            "subject-level stratified percentile bootstrap",
    }

    (
        out / "external_metrics.json"
    ).write_text(
        json.dumps(result, indent=2)
    )

    print(
        "\n======================================"
    )
    print(
        "FROZEN ROI-RF EXTERNAL VALIDATION"
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

        print(
            f"{k:20s} "
            f"{metrics[k]:.4f} "
            f"[{cis[k]['lower_95']:.4f}, "
            f"{cis[k]['upper_95']:.4f}]"
        )

    print("\nConfusion matrix:")
    print(
        "TN:", metrics["tn"],
        "FP:", metrics["fp"],
        "FN:", metrics["fn"],
        "TP:", metrics["tp"],
    )

    print(
        "\nExternal SZ prevalence / "
        "random PR-AUC baseline:",
        f"{y.mean():.4f}",
    )


if __name__ == "__main__":
    main()
