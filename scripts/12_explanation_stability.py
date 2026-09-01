#!/usr/bin/env python3

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def safe_spearman(a, b):
    a, b = a.align(b, join="inner")
    if len(a) < 3:
        return np.nan
    return float(spearmanr(a.values, b.values).statistic)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--node-attributions",
        required=True
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=10
    )

    p.add_argument(
        "--out",
        required=True
    )

    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.node_attributions)

    required = {
        "split", "repeat", "fold",
        "subject_id", "y", "roi",
        "feature", "importance"
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    print("Rows:", len(df))
    print("Subjects:", df.subject_id.nunique())
    print("Splits:", df["split"].nunique())
    print("Repeats:", sorted(df["repeat"].unique()))
    print("Top-k:", a.top_k)

    # ---------------------------------------------------------
    # 1. Collapse the five morphometric features into
    #    one ROI importance per held-out subject/model instance.
    # ---------------------------------------------------------

    instance_cols = [
        "split",
        "repeat",
        "fold",
        "subject_id",
        "y"
    ]

    optional_cols = [
        "score",
        "pred",
        "correct"
    ]

    for c in optional_cols:
        if c in df.columns:
            instance_cols.append(c)

    roi = (
        df.groupby(
            instance_cols + ["roi"],
            as_index=False
        )["importance"]
        .mean()
    )

    # GNNExplainer masks should normally be non-negative.
    print(
        "ROI importance range:",
        float(roi.importance.min()),
        "to",
        float(roi.importance.max())
    )

    # ---------------------------------------------------------
    # Normalize within each subject/model instance.
    # This makes group averaging less sensitive to arbitrary
    # mask-scale differences between independently fitted models.
    # ---------------------------------------------------------

    base_instance = [
        "split",
        "repeat",
        "fold",
        "subject_id",
        "y"
    ]

    # Preserve prediction metadata through all downstream
    # aggregation tables so TP/TN analyses remain available.
    for c in ["score", "pred", "correct"]:
        if c in roi.columns:
            base_instance.append(c)

    totals = (
        roi.groupby(base_instance)["importance"]
        .transform("sum")
    )

    totals = totals.replace(0, np.nan)

    roi["importance_share"] = (
        roi["importance"] / totals
    )

    roi["roi_rank"] = (
        roi.groupby(base_instance)["importance"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    roi.to_csv(
        out / "roi_instance_importance.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 2. Create EXACT top-k ROI selections per
    #    subject/model instance.
    # ---------------------------------------------------------

    top_rows = []

    for key, g in roi.groupby(base_instance):

        g = g.sort_values(
            ["importance", "roi"],
            ascending=[False, True]
        )

        top = g.head(a.top_k)

        meta = dict(
            zip(base_instance, key)
        )

        for rank_pos, row in enumerate(
            top.itertuples(),
            start=1
        ):
            top_rows.append({
                **meta,
                "roi": row.roi,
                "topk_rank": rank_pos,
                "importance": row.importance,
                "importance_share":
                    row.importance_share
            })

    tops = pd.DataFrame(top_rows)

    tops.to_csv(
        out / "subject_repeat_topk_rois.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 3. SAME-SUBJECT reproducibility across the three repeats.
    # Each subject should have one held-out explanation/repeat.
    # ---------------------------------------------------------

    subject_stability = []

    for sid, sg in roi.groupby("subject_id"):

        repeats = sorted(
            sg["repeat"].unique()
        )

        for r1, r2 in combinations(repeats, 2):

            g1 = sg[sg["repeat"] == r1]
            g2 = sg[sg["repeat"] == r2]

            s1 = (
                g1.set_index("roi")
                  ["importance_share"]
            )

            s2 = (
                g2.set_index("roi")
                  ["importance_share"]
            )

            t1 = (
                tops[
                    (tops.subject_id == sid) &
                    (tops["repeat"] == r1)
                ]["roi"]
                .tolist()
            )

            t2 = (
                tops[
                    (tops.subject_id == sid) &
                    (tops["repeat"] == r2)
                ]["roi"]
                .tolist()
            )

            subject_stability.append({
                "subject_id": sid,
                "y": int(sg["y"].iloc[0]),
                "repeat_a": int(r1),
                "repeat_b": int(r2),
                "jaccard_topk":
                    jaccard(t1, t2),
                "spearman_roi_rank":
                    safe_spearman(s1, s2)
            })

    subject_stability = pd.DataFrame(
        subject_stability
    )

    subject_stability.to_csv(
        out / "subject_repeat_stability.csv",
        index=False
    )

    subject_stability_summary = (
        subject_stability
        .groupby("y")
        .agg(
            n_pairs=("subject_id", "size"),
            mean_jaccard=(
                "jaccard_topk", "mean"
            ),
            median_jaccard=(
                "jaccard_topk", "median"
            ),
            mean_spearman=(
                "spearman_roi_rank", "mean"
            ),
            median_spearman=(
                "spearman_roi_rank", "median"
            )
        )
        .reset_index()
    )

    subject_stability_summary.to_csv(
        out /
        "subject_repeat_stability_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 4. Define analysis groups.
    #
    # ALL_SZ   = all true schizophrenia subjects.
    # ALL_CTRL = all true healthy controls.
    # TP       = correctly classified schizophrenia instances.
    # TN       = correctly classified control instances.
    #
    # ALL_SZ should be the main disease-group stability analysis.
    # TP is useful as a sensitivity analysis.
    # ---------------------------------------------------------

    group_specs = {
        "ALL_SZ":
            lambda x: x["y"] == 1,

        "ALL_CTRL":
            lambda x: x["y"] == 0,
    }

    if "correct" in df.columns:

        group_specs["TP"] = (
            lambda x:
            (x["y"] == 1) &
            (x["correct"] == 1)
        )

        group_specs["TN"] = (
            lambda x:
            (x["y"] == 0) &
            (x["correct"] == 1)
        )

    # ---------------------------------------------------------
    # 5. ROI selection frequency by repeat and group.
    # ---------------------------------------------------------

    freq_repeat_rows = []
    freq_overall_rows = []
    group_profile_rows = []

    for group_name, mask_fn in group_specs.items():

        roi_g = roi[
            mask_fn(roi)
        ].copy()

        tops_g = tops[
            mask_fn(tops)
        ].copy()

        if roi_g.empty:
            continue

        # Number of held-out subject/model instances
        # available for this group.
        all_instances = (
            roi_g[base_instance]
            .drop_duplicates()
        )

        # ------------------------------
        # Per-repeat selection frequency
        # ------------------------------

        for repeat, rg in roi_g.groupby("repeat"):

            tg = tops_g[
                tops_g["repeat"] == repeat
            ]

            n_instances = (
                rg[base_instance]
                .drop_duplicates()
                .shape[0]
            )

            counts = (
                tg["roi"]
                .value_counts()
            )

            profile = (
                rg.groupby("roi")
                .agg(
                    mean_importance_share=(
                        "importance_share", "mean"
                    ),
                    mean_rank=(
                        "roi_rank", "mean"
                    )
                )
                .reset_index()
            )

            for row in profile.itertuples():

                count = int(
                    counts.get(row.roi, 0)
                )

                freq_repeat_rows.append({
                    "group": group_name,
                    "repeat": int(repeat),
                    "roi": row.roi,
                    "topk_count": count,
                    "n_instances":
                        n_instances,
                    "selection_frequency":
                        count /
                        max(n_instances, 1),
                    "mean_importance_share":
                        row.mean_importance_share,
                    "mean_rank":
                        row.mean_rank
                })

                group_profile_rows.append({
                    "group": group_name,
                    "repeat": int(repeat),
                    "roi": row.roi,
                    "mean_importance_share":
                        row.mean_importance_share,
                    "mean_rank":
                        row.mean_rank
                })

        # ------------------------------
        # Overall selection frequency
        # across all three repeats.
        # ------------------------------

        n_instances = (
            all_instances.shape[0]
        )

        counts = (
            tops_g["roi"]
            .value_counts()
        )

        profile = (
            roi_g.groupby("roi")
            .agg(
                mean_importance_share=(
                    "importance_share", "mean"
                ),
                mean_rank=(
                    "roi_rank", "mean"
                )
            )
            .reset_index()
        )

        for row in profile.itertuples():

            count = int(
                counts.get(row.roi, 0)
            )

            freq_overall_rows.append({
                "group": group_name,
                "roi": row.roi,
                "topk_count": count,
                "n_instances":
                    n_instances,
                "selection_frequency":
                    count /
                    max(n_instances, 1),
                "mean_importance_share":
                    row.mean_importance_share,
                "mean_rank":
                    row.mean_rank
            })

    freq_repeat = pd.DataFrame(
        freq_repeat_rows
    )

    freq_overall = pd.DataFrame(
        freq_overall_rows
    )

    profiles = pd.DataFrame(
        group_profile_rows
    )

    freq_repeat.to_csv(
        out /
        "roi_selection_frequency_by_repeat.csv",
        index=False
    )

    freq_overall.to_csv(
        out /
        "roi_selection_frequency_overall.csv",
        index=False
    )

    profiles.to_csv(
        out /
        "group_repeat_roi_profiles.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 6. GROUP-LEVEL repeat stability.
    #
    # This is one of the most important outputs:
    # Does the schizophrenia ROI ranking reproduce
    # across repeat 0, 1, and 2?
    # ---------------------------------------------------------

    group_stability_rows = []

    for group_name, gg in profiles.groupby("group"):

        repeats = sorted(
            gg["repeat"].unique()
        )

        for r1, r2 in combinations(repeats, 2):

            g1 = gg[gg["repeat"] == r1]
            g2 = gg[gg["repeat"] == r2]

            s1 = (
                g1.set_index("roi")
                  ["mean_importance_share"]
            )

            s2 = (
                g2.set_index("roi")
                  ["mean_importance_share"]
            )

            top1 = (
                g1.sort_values(
                    "mean_importance_share",
                    ascending=False
                )
                .head(a.top_k)["roi"]
                .tolist()
            )

            top2 = (
                g2.sort_values(
                    "mean_importance_share",
                    ascending=False
                )
                .head(a.top_k)["roi"]
                .tolist()
            )

            group_stability_rows.append({
                "group": group_name,
                "repeat_a": int(r1),
                "repeat_b": int(r2),
                "jaccard_topk":
                    jaccard(top1, top2),
                "spearman_roi_rank":
                    safe_spearman(s1, s2)
            })

    group_stability = pd.DataFrame(
        group_stability_rows
    )

    group_stability.to_csv(
        out /
        "group_repeat_stability.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 7. Morphometric FEATURE importance.
    #
    # Average over ROIs within each subject/model instance,
    # then normalize across the five features.
    # ---------------------------------------------------------

    feature_instance = (
        df.groupby(
            base_instance + ["feature"],
            as_index=False
        )["importance"]
        .mean()
    )

    feature_totals = (
        feature_instance
        .groupby(base_instance)["importance"]
        .transform("sum")
        .replace(0, np.nan)
    )

    feature_instance[
        "importance_share"
    ] = (
        feature_instance["importance"] /
        feature_totals
    )

    feature_rows = []

    for group_name, mask_fn in group_specs.items():

        fg = feature_instance[
            mask_fn(feature_instance)
        ]

        if fg.empty:
            continue

        for feature, g in fg.groupby("feature"):

            feature_rows.append({
                "group": group_name,
                "feature": feature,
                "mean_importance_share":
                    g["importance_share"].mean(),
                "std_importance_share":
                    g["importance_share"].std()
            })

    feature_summary = pd.DataFrame(
        feature_rows
    )

    feature_summary.to_csv(
        out /
        "feature_importance_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. Exploratory SZ-vs-control frequency difference.
    #
    # Important: this is NOT direction of biological effect.
    # It only asks whether an ROI is selected more frequently
    # in explanations of true SZ vs true control subjects.
    # ---------------------------------------------------------

    sz = (
        freq_overall[
            freq_overall.group == "ALL_SZ"
        ][
            [
                "roi",
                "selection_frequency",
                "mean_importance_share"
            ]
        ]
        .rename(columns={
            "selection_frequency":
                "sz_selection_frequency",
            "mean_importance_share":
                "sz_mean_importance_share"
        })
    )

    ctrl = (
        freq_overall[
            freq_overall.group == "ALL_CTRL"
        ][
            [
                "roi",
                "selection_frequency",
                "mean_importance_share"
            ]
        ]
        .rename(columns={
            "selection_frequency":
                "ctrl_selection_frequency",
            "mean_importance_share":
                "ctrl_mean_importance_share"
        })
    )

    contrast = sz.merge(
        ctrl,
        on="roi",
        how="inner"
    )

    contrast[
        "selection_frequency_difference"
    ] = (
        contrast["sz_selection_frequency"] -
        contrast["ctrl_selection_frequency"]
    )

    contrast[
        "importance_share_difference"
    ] = (
        contrast["sz_mean_importance_share"] -
        contrast["ctrl_mean_importance_share"]
    )

    contrast = contrast.sort_values(
        "selection_frequency_difference",
        ascending=False
    )

    contrast.to_csv(
        out /
        "sz_control_explanation_contrast.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Console summaries
    # ---------------------------------------------------------

    print("\n========================================")
    print("SAME-SUBJECT REPEAT STABILITY")
    print("========================================")
    print(
        subject_stability_summary
        .to_string(index=False)
    )

    print("\n========================================")
    print("GROUP-LEVEL REPEAT STABILITY")
    print("========================================")
    print(
        group_stability
        .to_string(index=False)
    )

    print("\n========================================")
    print(
        f"TOP {a.top_k} SCHIZOPHRENIA ROIs"
    )
    print("by selection frequency")
    print("========================================")

    top_sz = (
        freq_overall[
            freq_overall.group == "ALL_SZ"
        ]
        .sort_values(
            [
                "selection_frequency",
                "mean_importance_share"
            ],
            ascending=False
        )
        .head(a.top_k)
    )

    print(
        top_sz[
            [
                "roi",
                "selection_frequency",
                "mean_importance_share",
                "mean_rank"
            ]
        ].to_string(index=False)
    )

    print("\n========================================")
    print("MORPHOMETRIC FEATURE IMPORTANCE — SZ")
    print("========================================")

    print(
        feature_summary[
            feature_summary.group == "ALL_SZ"
        ]
        .sort_values(
            "mean_importance_share",
            ascending=False
        )
        .to_string(index=False)
    )

    print()
    print(
        "Wrote stability outputs to:",
        out
    )


if __name__ == "__main__":
    main()
