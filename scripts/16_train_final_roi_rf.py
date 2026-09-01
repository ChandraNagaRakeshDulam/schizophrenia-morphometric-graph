#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # INTERNAL DEVELOPMENT DATA ONLY
    # -----------------------------
    d = np.load(a.dataset, allow_pickle=True)

    y = d["y"].astype(int)

    # Identical ROI representation used by 08_ml_nested_cv.py
    X = d["X_nodes"].astype(np.float32).reshape(len(y), -1)

    roi_names = d["roi_names"].astype(str)
    feature_names = d["feature_names"].astype(str)

    assert X.shape == (175, 340)
    assert len(y) == 175
    assert int(y.sum()) == 50
    assert int((y == 0).sum()) == 125

    print("Development subjects:", len(y))
    print("Cases:", int(y.sum()))
    print("Controls:", int((y == 0).sum()))
    print("Features:", X.shape[1])

    # Same estimator and grid as original nested-CV baseline.
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=a.seed,
    )

    grid = {
        "n_estimators": [300, 600],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", 0.5],
    }

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=a.seed,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    print("\nRunning development-only RF model selection...")
    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)

    keep = [
        "param_n_estimators",
        "param_max_depth",
        "param_max_features",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]

    summary = (
        results[keep]
        .sort_values(
            ["rank_test_score", "mean_test_score"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        out / "cv_summary.csv",
        index=False,
    )

    print("\n======================================")
    print("ROI-RF DEVELOPMENT-ONLY CV SUMMARY")
    print("======================================")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSELECTED PARAMETERS")
    print("------------------------------")
    print(search.best_params_)
    print(
        "Mean development CV ROC-AUC:",
        f"{search.best_score_:.6f}"
    )

    # search.best_estimator_ has already been refit on ALL 175 subjects.
    frozen = {
        "model": search.best_estimator_,
        "representation": "roi",
        "model_kind": "rf",
        "best_params": search.best_params_,
        "selected_cv_mean_auc": float(search.best_score_),
        "development_dataset": str(a.dataset),
        "n_train": int(len(y)),
        "n_cases": int(y.sum()),
        "n_controls": int((y == 0).sum()),
        "roi_names": roi_names.tolist(),
        "feature_names": feature_names.tolist(),
        "input_shape": [68, 5],
        "flattened_features": 340,
        "class_weight": "balanced",
        "random_state": int(a.seed),
        "threshold": 0.5,
        "external_data_used_for_selection": False,
        "selection_cv": "RepeatedStratifiedKFold 5x3",
        "selection_metric": "ROC-AUC",
    }

    model_path = out / "final_roi_rf_ds000030.joblib"

    joblib.dump(
        frozen,
        model_path,
        compress=3,
    )

    selection = {
        "model": "RandomForestClassifier",
        "representation": "DK68 regional morphology",
        "features": 340,
        "best_params": search.best_params_,
        "development_cv_mean_auc": float(search.best_score_),
        "development_dataset": str(a.dataset),
        "external_dataset_used_for_selection": False,
        "threshold": 0.5,
    }

    (out / "selection.json").write_text(
        json.dumps(selection, indent=2)
    )

    # SHA-256 freeze
    h = hashlib.sha256()

    with open(model_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    sha = h.hexdigest()

    (out / "final_model.sha256").write_text(
        f"{sha}  {model_path.name}\n"
    )

    print("\n======================================")
    print("FINAL ROI-RF FROZEN")
    print("======================================")
    print("Model:", model_path)
    print("SHA-256:", sha)

    print(
        "\nDo not change RF parameters after viewing "
        "ds004302 results."
    )


if __name__ == "__main__":
    main()
