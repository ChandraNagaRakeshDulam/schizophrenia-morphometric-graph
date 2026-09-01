#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    boots = np.empty(n_boot)

    for b in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        boots[b] = sample.mean()

    return (
        values.mean(),
        np.percentile(boots, 2.5),
        np.percentile(boots, 97.5)
    )


def stratified_bootstrap(df, value_col,
                         n_boot=10000,
                         seed=42):

    rng = np.random.default_rng(seed)

    groups = {
        y: g[value_col].to_numpy()
        for y, g in df.groupby("y")
    }

    boots = []

    for _ in range(n_boot):

        sampled = []

        for y, vals in groups.items():
            sampled.append(
                rng.choice(
                    vals,
                    size=len(vals),
                    replace=True
                )
            )

        boots.append(
            np.concatenate(sampled).mean()
        )

    boots = np.asarray(boots)

    return (
        df[value_col].mean(),
        np.percentile(boots, 2.5),
        np.percentile(boots, 97.5)
    )


def print_result(name, df, col,
                 stratified=False):

    if stratified:
        mean, lo, hi = stratified_bootstrap(
            df, col
        )
    else:
        mean, lo, hi = bootstrap_mean_ci(
            df[col]
        )

    print("\n" + name)
    print("-" * len(name))
    print("Subjects:", len(df))
    print("Mean:", mean)
    print("95% bootstrap CI:",
          f"[{lo:.6f}, {hi:.6f}]")
    print(
        "Median:",
        df[col].median()
    )
    print(
        "Subjects with positive advantage:",
        (df[col] > 0).mean()
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True
    )

    a = p.parse_args()

    df = pd.read_csv(a.input)

    print("Rows:", len(df))
    print(
        "Unique subjects:",
        df.subject_id.nunique()
    )

    # -------------------------------------------------
    # All instances:
    # average the three CV-repeat estimates first.
    # -------------------------------------------------

    subject = (
        df.groupby(
            ["subject_id", "y"],
            as_index=False
        )
        .agg(
            top_drop=(
                "topk_predicted_class_conf_drop",
                "mean"
            ),
            random_drop=(
                "random_mean_predicted_class_conf_drop",
                "mean"
            ),
            faithfulness_advantage=(
                "faithfulness_advantage",
                "mean"
            ),
            n_instances=(
                "split", "size"
            )
        )
    )

    print_result(
        "ALL SUBJECTS — FAITHFULNESS ADVANTAGE",
        subject,
        "faithfulness_advantage",
        stratified=True
    )

    print("\nAll-subject mean drops")
    print(
        "Top-10:",
        subject.top_drop.mean()
    )
    print(
        "Random:",
        subject.random_drop.mean()
    )

    # -------------------------------------------------
    # True SZ subjects, regardless of prediction.
    # Predicted-class faithfulness.
    # -------------------------------------------------

    sz_subject = subject[
        subject.y == 1
    ].copy()

    print_result(
        "TRUE SZ SUBJECTS — PREDICTED-CLASS FAITHFULNESS",
        sz_subject,
        "faithfulness_advantage"
    )

    # -------------------------------------------------
    # True-positive SZ instances only.
    #
    # A subject can be TP in 1, 2, or 3 outer repeats.
    # Collapse TP runs to one estimate per subject.
    # -------------------------------------------------

    tp = df[
        (df.y == 1) &
        (df.pred == 1)
    ].copy()

    tp[
        "sz_faithfulness_advantage"
    ] = (
        tp["topk_sz_score_drop"] -
        tp["random_mean_sz_score_drop"]
    )

    tp_subject = (
        tp.groupby(
            ["subject_id", "y"],
            as_index=False
        )
        .agg(
            top_sz_drop=(
                "topk_sz_score_drop",
                "mean"
            ),
            random_sz_drop=(
                "random_mean_sz_score_drop",
                "mean"
            ),
            sz_faithfulness_advantage=(
                "sz_faithfulness_advantage",
                "mean"
            ),
            n_tp_instances=(
                "split", "size"
            )
        )
    )

    print_result(
        "TRUE-POSITIVE SZ SUBJECTS — SZ PROBABILITY FAITHFULNESS",
        tp_subject,
        "sz_faithfulness_advantage"
    )

    print("\nTP-SZ mean probability drops")
    print(
        "Top-10:",
        tp_subject.top_sz_drop.mean()
    )
    print(
        "Random:",
        tp_subject.random_sz_drop.mean()
    )

    print(
        "\nTP repeat coverage:"
    )

    print(
        tp_subject[
            "n_tp_instances"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()
