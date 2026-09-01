#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def normalize_instance_importance(df, instance_cols):
    """
    Normalize GNNExplainer node-feature importance within each
    subject/model instance so each explanation sums to 1.
    """
    x = df.copy()

    totals = (
        x.groupby(instance_cols)["importance"]
        .transform("sum")
    )

    totals = totals.replace(0, np.nan)

    x["importance_norm"] = (
        x["importance"] / totals
    ).fillna(0.0)

    return x


def make_subject_roi_profiles(df, internal):
    """
    ROI importance per subject.

    Internal subjects have 3 held-out explanation instances.
    We average repeats within subject FIRST so no internal
    subject receives triple weight.
    """

    instance_cols = (
        ["split", "subject_id"]
        if internal
        else ["subject_id"]
    )

    x = normalize_instance_importance(
        df,
        instance_cols
    )

    # Sum five features -> ROI importance per explanation.
    roi_instance = (
        x.groupby(
            instance_cols + ["y", "roi"],
            as_index=False
        )
        .agg(
            roi_importance=(
                "importance_norm",
                "sum"
            )
        )
    )

    # Average repeated internal explanations within subject.
    subject_roi = (
        roi_instance
        .groupby(
            ["subject_id", "y", "roi"],
            as_index=False
        )
        .agg(
            roi_importance=(
                "roi_importance",
                "mean"
            )
        )
    )

    # Re-normalize each subject after averaging.
    subject_roi["roi_importance"] = (
        subject_roi["roi_importance"]
        /
        subject_roi.groupby(
            "subject_id"
        )["roi_importance"].transform("sum")
    )

    return subject_roi


def make_subject_feature_profiles(df, internal):

    instance_cols = (
        ["split", "subject_id"]
        if internal
        else ["subject_id"]
    )

    x = normalize_instance_importance(
        df,
        instance_cols
    )

    feat_instance = (
        x.groupby(
            instance_cols + ["y", "feature"],
            as_index=False
        )
        .agg(
            feature_importance=(
                "importance_norm",
                "sum"
            )
        )
    )

    subject_feat = (
        feat_instance
        .groupby(
            ["subject_id", "y", "feature"],
            as_index=False
        )
        .agg(
            feature_importance=(
                "feature_importance",
                "mean"
            )
        )
    )

    subject_feat["feature_importance"] = (
        subject_feat["feature_importance"]
        /
        subject_feat.groupby(
            "subject_id"
        )["feature_importance"].transform("sum")
    )

    return subject_feat


def group_profile(df, value_col, y_value):

    return (
        df[df["y"] == y_value]
        .groupby(
            df.columns[
                -2 if value_col == "roi_importance"
                else -2
            ]
        )
    )


def roi_group_mean(subject_roi, y_value):

    return (
        subject_roi[
            subject_roi.y == y_value
        ]
        .groupby(
            "roi",
            as_index=False
        )
        .agg(
            importance=(
                "roi_importance",
                "mean"
            )
        )
        .sort_values(
            "roi"
        )
        .reset_index(drop=True)
    )


def feature_group_mean(subject_feat, y_value):

    return (
        subject_feat[
            subject_feat.y == y_value
        ]
        .groupby(
            "feature",
            as_index=False
        )
        .agg(
            importance=(
                "feature_importance",
                "mean"
            )
        )
        .sort_values(
            "feature"
        )
        .reset_index(drop=True)
    )


def topk_set(profile, key, k):

    return set(
        profile.sort_values(
            "importance",
            ascending=False
        )
        .head(k)[key]
        .tolist()
    )


def jaccard(a, b):

    if not (a or b):
        return np.nan

    return len(a & b) / len(a | b)


def compare_profiles(
    internal_profile,
    external_profile,
    key,
    k
):

    m = internal_profile.merge(
        external_profile,
        on=key,
        suffixes=("_internal", "_external"),
        validate="one_to_one"
    )

    rho = spearmanr(
        m["importance_internal"],
        m["importance_external"]
    ).statistic

    a = topk_set(
        internal_profile,
        key,
        k
    )

    b = topk_set(
        external_profile,
        key,
        k
    )

    jac = jaccard(a, b)

    overlap = sorted(a & b)

    return m, float(rho), float(jac), overlap


def bootstrap_roi_repro(
    int_subject_roi,
    ext_subject_roi,
    y_value,
    k,
    n_boot,
    seed
):

    rng = np.random.default_rng(seed)

    i_subjects = (
        int_subject_roi.loc[
            int_subject_roi.y == y_value,
            "subject_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    e_subjects = (
        ext_subject_roi.loc[
            ext_subject_roi.y == y_value,
            "subject_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    rhos = []
    jacs = []

    for _ in range(n_boot):

        ib = rng.choice(
            i_subjects,
            size=len(i_subjects),
            replace=True
        )

        eb = rng.choice(
            e_subjects,
            size=len(e_subjects),
            replace=True
        )

        # Preserve multiplicity after bootstrap resampling.
        i_parts = []

        for n, sid in enumerate(ib):
            q = int_subject_roi[
                int_subject_roi.subject_id == sid
            ].copy()
            q["boot_id"] = n
            i_parts.append(q)

        e_parts = []

        for n, sid in enumerate(eb):
            q = ext_subject_roi[
                ext_subject_roi.subject_id == sid
            ].copy()
            q["boot_id"] = n
            e_parts.append(q)

        ix = pd.concat(
            i_parts,
            ignore_index=True
        )

        ex = pd.concat(
            e_parts,
            ignore_index=True
        )

        ip = (
            ix.groupby("roi", as_index=False)
            .agg(
                importance=(
                    "roi_importance",
                    "mean"
                )
            )
        )

        ep = (
            ex.groupby("roi", as_index=False)
            .agg(
                importance=(
                    "roi_importance",
                    "mean"
                )
            )
        )

        m = ip.merge(
            ep,
            on="roi",
            suffixes=("_internal", "_external")
        )

        rho = spearmanr(
            m.importance_internal,
            m.importance_external
        ).statistic

        jac = jaccard(
            topk_set(ip, "roi", k),
            topk_set(ep, "roi", k)
        )

        rhos.append(rho)
        jacs.append(jac)

    def ci(x):

        x = np.asarray(
            x,
            dtype=float
        )

        return {
            "mean":
                float(np.nanmean(x)),

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

    return {
        "spearman":
            ci(rhos),

        "topk_jaccard":
            ci(jacs),
    }


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--internal",
        required=True
    )

    p.add_argument(
        "--external",
        required=True
    )

    p.add_argument(
        "--out",
        required=True
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=10
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=5000
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    a = p.parse_args()

    out = Path(a.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    i = pd.read_csv(
        a.internal
    )

    e = pd.read_csv(
        a.external
    )

    print("Internal rows:", len(i))
    print("External rows:", len(e))

    assert set(i.roi.unique()) == set(e.roi.unique())
    assert set(i.feature.unique()) == set(e.feature.unique())

    int_roi = make_subject_roi_profiles(
        i,
        internal=True
    )

    ext_roi = make_subject_roi_profiles(
        e,
        internal=False
    )

    int_feat = make_subject_feature_profiles(
        i,
        internal=True
    )

    ext_feat = make_subject_feature_profiles(
        e,
        internal=False
    )

    int_roi.to_csv(
        out / "internal_subject_roi_profiles.csv",
        index=False
    )

    ext_roi.to_csv(
        out / "external_subject_roi_profiles.csv",
        index=False
    )

    results = {}

    labels = {
        "CONTROL": 0,
        "SZ": 1
    }

    all_roi_tables = []

    for label, yy in labels.items():

        ip = roi_group_mean(
            int_roi,
            yy
        )

        ep = roi_group_mean(
            ext_roi,
            yy
        )

        comparison, rho, jac, overlap = (
            compare_profiles(
                ip,
                ep,
                "roi",
                a.top_k
            )
        )

        comparison["group"] = label

        all_roi_tables.append(
            comparison
        )

        boot = bootstrap_roi_repro(
            int_roi,
            ext_roi,
            yy,
            a.top_k,
            a.bootstrap,
            a.seed + yy
        )

        results[label] = {
            "n_internal":
                int(
                    int_roi.loc[
                        int_roi.y == yy,
                        "subject_id"
                    ].nunique()
                ),

            "n_external":
                int(
                    ext_roi.loc[
                        ext_roi.y == yy,
                        "subject_id"
                    ].nunique()
                ),

            "roi_spearman":
                rho,

            f"top_{a.top_k}_jaccard":
                jac,

            f"top_{a.top_k}_overlap":
                overlap,

            "bootstrap":
                boot,
        }

        print()
        print(
            "======================================"
        )

        print(
            f"{label} CROSS-COHORT ROI REPRODUCIBILITY"
        )

        print(
            "======================================"
        )

        print(
            "Internal subjects:",
            results[label]["n_internal"]
        )

        print(
            "External subjects:",
            results[label]["n_external"]
        )

        print(
            "ROI Spearman:",
            f"{rho:.4f}"
        )

        print(
            f"Top-{a.top_k} Jaccard:",
            f"{jac:.4f}"
        )

        print(
            f"Top-{a.top_k} overlap:"
        )

        for roi in overlap:
            print(" ", roi)

        print(
            "Bootstrap Spearman 95% CI:",
            f"[{boot['spearman']['lower_95']:.4f}, "
            f"{boot['spearman']['upper_95']:.4f}]"
        )

        print(
            "Bootstrap Jaccard 95% CI:",
            f"[{boot['topk_jaccard']['lower_95']:.4f}, "
            f"{boot['topk_jaccard']['upper_95']:.4f}]"
        )

    pd.concat(
        all_roi_tables,
        ignore_index=True
    ).to_csv(
        out / "roi_cross_cohort_comparison.csv",
        index=False
    )

    # ----------------------------------------
    # Feature-level reproducibility
    # ----------------------------------------

    feature_results = {}

    for label, yy in labels.items():

        ip = feature_group_mean(
            int_feat,
            yy
        )

        ep = feature_group_mean(
            ext_feat,
            yy
        )

        m = ip.merge(
            ep,
            on="feature",
            suffixes=("_internal", "_external")
        )

        rho = spearmanr(
            m.importance_internal,
            m.importance_external
        ).statistic

        m["group"] = label

        m.to_csv(
            out /
            f"{label.lower()}_feature_comparison.csv",
            index=False
        )

        feature_results[label] = {
            "spearman":
                float(rho),

            "internal_ranking":
                ip.sort_values(
                    "importance",
                    ascending=False
                ).to_dict("records"),

            "external_ranking":
                ep.sort_values(
                    "importance",
                    ascending=False
                ).to_dict("records"),
        }

        print()
        print(
            f"{label} FEATURE SPEARMAN:",
            f"{rho:.4f}"
        )

        print("\nInternal:")
        print(
            ip.sort_values(
                "importance",
                ascending=False
            ).to_string(index=False)
        )

        print("\nExternal:")
        print(
            ep.sort_values(
                "importance",
                ascending=False
            ).to_string(index=False)
        )

    results["feature_reproducibility"] = (
        feature_results
    )

    (
        out /
        "cross_cohort_reproducibility.json"
    ).write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    print()
    print(
        "======================================"
    )

    print(
        "CROSS-COHORT ANALYSIS COMPLETE"
    )

    print(
        "======================================"
    )

    print("Output:", out)


if __name__ == "__main__":
    main()
