#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix

def upper_edges(A):
    iu = np.triu_indices(A.shape[1], k=1)
    return A[:, iu[0], iu[1]]

def get_X(d, rep):
    if rep == "roi":
        return d["X_nodes"].reshape(len(d["y"]), -1)
    if rep == "msn_edges":
        return upper_edges(d["A_msn"])
    if rep == "mind_edges":
        return upper_edges(d["A_mind"])
    raise ValueError(rep)

def model_and_grid(name):
    if name == "lr":
        est = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "svm":
        est = Pipeline([("scale", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
        grid = {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"], "clf__gamma": ["scale", 0.01, 0.1]}
    elif name == "rf":
        est = RandomForestClassifier(class_weight="balanced", random_state=42)
        grid = {"n_estimators": [300, 600], "max_depth": [None, 5, 10], "max_features": ["sqrt", 0.5]}
    else:
        raise ValueError("Use lr, svm, or rf in this starter.")
    return est, grid

def scores(est, X):
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]

def metrics(y, s, threshold=0.0):
    # SVC decision_function threshold is 0; LR predict_proba would use .predict below.
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--representation", choices=["roi","msn_edges","mind_edges"], required=True)
    p.add_argument("--model", choices=["lr","svm","rf"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    ids, y = d["ids"].astype(str), d["y"].astype(int)
    X = get_X(d, a.representation)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    base, grid = model_and_grid(a.model)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=a.seed)

    rows = []
    for split_id, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=a.seed + split_id)
        search = GridSearchCV(clone(base), grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X[tr], y[tr])
        est = search.best_estimator_
        s = scores(est, X[te])
        pred = est.predict(X[te])
        for j, idx in enumerate(te):
            rows.append({
                "split": split_id, "subject_id": ids[idx], "y": y[idx],
                "score": float(s[j]), "pred": int(pred[j]),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out / "outer_predictions.csv", index=False)

    # Each subject is predicted once per repeat. Average scores across repeats before final pooled metrics.
    agg = pred_df.groupby(["subject_id","y"], as_index=False).agg(score=("score","mean"))
    # Use 0 for decision-function models. For RF probabilities, use 0.5.
    threshold = 0.5 if a.model == "rf" else 0.0
    m = metrics(agg["y"].to_numpy(), agg["score"].to_numpy(), threshold)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
