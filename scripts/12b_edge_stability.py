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


def edge_name(a, b):
    return " -- ".join(sorted([str(a), str(b)]))


def safe_spearman(a, b):
    a, b = a.align(b, join="inner")
    if len(a) < 3:
        return np.nan
    return float(spearmanr(a.values, b.values).statistic)


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--edge-attributions", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.edge_attributions)

    print("Directed rows:", len(df))
    print("Subjects:", df.subject_id.nunique())
    print("Splits:", df["split"].nunique())

    # -------------------------------------------
    # Convert directed duplicates into
    # one undirected edge per subject/model.
    # -------------------------------------------

    df["edge"] = [
        edge_name(u, v)
        for u, v in zip(df.source, df.target)
    ]

    base = [
        "split", "repeat", "fold",
        "subject_id", "y"
    ]

    for c in ["score", "pred", "correct", "density", "hidden"]:
        if c in df.columns:
            base.append(c)

    edge = (
        df.groupby(
            base + ["edge"],
            as_index=False
        )
        .agg(
            importance=("importance", "mean"),
            original_edge_weight=("original_edge_weight", "mean")
        )
    )

    print("Undirected rows:", len(edge))

    # Normalize explanation magnitude within each graph.
    totals = (
        edge.groupby(base)["importance"]
        .transform("sum")
        .replace(0, np.nan)
    )

    edge["importance_share"] = (
        edge.importance / totals
    )

    edge["edge_rank"] = (
        edge.groupby(base)["importance"]
        .rank(ascending=False, method="average")
    )

    edge.to_csv(
        out / "edge_instance_importance.csv",
        index=False
    )

    # -------------------------------------------
    # Exact top-k per subject-model instance
    # -------------------------------------------

    top_rows = []

    for key, g in edge.groupby(base):

        top = (
            g.sort_values(
                ["importance", "edge"],
                ascending=[False, True]
            )
            .head(a.top_k)
        )

        meta = dict(zip(base, key))

        for rank_pos, row in enumerate(
            top.itertuples(),
            start=1
        ):
            top_rows.append({
                **meta,
                "edge": row.edge,
                "topk_rank": rank_pos,
                "importance": row.importance,
                "importance_share": row.importance_share,
                "original_edge_weight": row.original_edge_weight
            })

    tops = pd.DataFrame(top_rows)

    tops.to_csv(
        out / "subject_repeat_topk_edges.csv",
        index=False
    )

    # -------------------------------------------
    # Same-subject stability across repeats
    # -------------------------------------------

    stab_rows = []

    for sid, sg in edge.groupby("subject_id"):

        repeats = sorted(sg["repeat"].unique())

        for r1, r2 in combinations(repeats, 2):

            g1 = sg[sg["repeat"] == r1]
            g2 = sg[sg["repeat"] == r2]

            s1 = g1.set_index("edge")["importance_share"]
            s2 = g2.set_index("edge")["importance_share"]

            t1 = tops[
                (tops.subject_id == sid) &
                (tops["repeat"] == r1)
            ]["edge"].tolist()

            t2 = tops[
                (tops.subject_id == sid) &
                (tops["repeat"] == r2)
            ]["edge"].tolist()

            stab_rows.append({
                "subject_id": sid,
                "y": int(sg.y.iloc[0]),
                "repeat_a": int(r1),
                "repeat_b": int(r2),
                "jaccard_topk": jaccard(t1, t2),
                "spearman_edge_rank": safe_spearman(s1, s2)
            })

    stab = pd.DataFrame(stab_rows)

    stab.to_csv(
        out / "subject_repeat_edge_stability.csv",
        index=False
    )

    # -------------------------------------------
    # SZ/control top-edge frequency
    # -------------------------------------------

    group_specs = {
        "ALL_SZ": lambda x: x.y == 1,
        "ALL_CTRL": lambda x: x.y == 0
    }

    overall_rows = []

    for group_name, fn in group_specs.items():

        eg = edge[fn(edge)]
        tg = tops[fn(tops)]

        n_instances = (
            eg[base]
            .drop_duplicates()
            .shape[0]
        )

        counts = tg.edge.value_counts()

        profile = (
            eg.groupby("edge")
            .agg(
                mean_importance_share=("importance_share", "mean"),
                mean_rank=("edge_rank", "mean"),
                mean_original_weight=("original_edge_weight", "mean")
            )
            .reset_index()
        )

        for row in profile.itertuples():

            count = int(
                counts.get(row.edge, 0)
            )

            overall_rows.append({
                "group": group_name,
                "edge": row.edge,
                "topk_count": count,
                "n_instances": n_instances,
                "selection_frequency":
                    count / max(n_instances, 1),
                "mean_importance_share":
                    row.mean_importance_share,
                "mean_rank":
                    row.mean_rank,
                "mean_original_weight":
                    row.mean_original_weight
            })

    overall = pd.DataFrame(overall_rows)

    overall.to_csv(
        out / "edge_selection_frequency_overall.csv",
        index=False
    )

    # -------------------------------------------
    # SZ-control contrast
    # -------------------------------------------

    sz = (
        overall[overall.group == "ALL_SZ"]
        [
            ["edge", "selection_frequency",
             "mean_importance_share"]
        ]
        .rename(columns={
            "selection_frequency": "sz_selection_frequency",
            "mean_importance_share": "sz_mean_importance_share"
        })
    )

    ctrl = (
        overall[overall.group == "ALL_CTRL"]
        [
            ["edge", "selection_frequency",
             "mean_importance_share"]
        ]
        .rename(columns={
            "selection_frequency": "ctrl_selection_frequency",
            "mean_importance_share": "ctrl_mean_importance_share"
        })
    )

    contrast = sz.merge(
        ctrl,
        on="edge",
        how="inner"
    )

    contrast["selection_frequency_difference"] = (
        contrast.sz_selection_frequency -
        contrast.ctrl_selection_frequency
    )

    contrast["importance_share_difference"] = (
        contrast.sz_mean_importance_share -
        contrast.ctrl_mean_importance_share
    )

    contrast = contrast.sort_values(
        "selection_frequency_difference",
        ascending=False
    )

    contrast.to_csv(
        out / "sz_control_edge_contrast.csv",
        index=False
    )

    # -------------------------------------------
    # Console output
    # -------------------------------------------

    print("\nTOP SCHIZOPHRENIA EDGES")
    print(
        overall[
            overall.group == "ALL_SZ"
        ]
        .sort_values(
            ["selection_frequency",
             "mean_importance_share"],
            ascending=False
        )
        .head(a.top_k)
        [
            [
                "edge",
                "selection_frequency",
                "mean_importance_share",
                "mean_original_weight"
            ]
        ]
        .to_string(index=False)
    )

    print("\nTOP SZ-ENRICHED EDGES")
    print(
        contrast.head(a.top_k)
        [
            [
                "edge",
                "sz_selection_frequency",
                "ctrl_selection_frequency",
                "selection_frequency_difference",
                "importance_share_difference"
            ]
        ]
        .to_string(index=False)
    )

    print("\nSAME-SUBJECT EDGE STABILITY")

    print(
        stab.groupby("y")
        .agg(
            mean_jaccard=("jaccard_topk", "mean"),
            median_jaccard=("jaccard_topk", "median"),
            mean_spearman=("spearman_edge_rank", "mean"),
            median_spearman=("spearman_edge_rank", "median")
        )
        .reset_index()
        .to_string(index=False)
    )

    print("\nWrote:", out)


if __name__ == "__main__":
    main()
