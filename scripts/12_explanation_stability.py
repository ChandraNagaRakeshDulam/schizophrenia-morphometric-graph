#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node-attributions", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    df = pd.read_csv(a.node_attributions)
    # Collapse features to ROI importance per subject.
    roi = df.groupby(["fold","subject_id","y","roi"], as_index=False)["importance"].mean()

    rankings = {}
    top_sets = {}
    for sid, g in roi.groupby("subject_id"):
        s = g.groupby("roi")["importance"].mean().sort_values(ascending=False)
        rankings[sid] = s
        top_sets[sid] = list(s.head(a.top_k).index)

    # Subject-level explanation reproducibility is not the same as fold stability,
    # because each held-out subject appears in only one outer fold here.
    # This file is mainly a template for repeated-CV/seed runs.
    rows = []
    common_subjects = sorted(rankings)
    for a_id, b_id in combinations(common_subjects, 2):
        ra, rb = rankings[a_id].align(rankings[b_id], join="inner")
        rho = spearmanr(ra.values, rb.values).statistic if len(ra) > 2 else np.nan
        rows.append({
            "subject_a": a_id, "subject_b": b_id,
            "jaccard_topk": jaccard(top_sets[a_id], top_sets[b_id]),
            "spearman_roi_rank": rho,
        })

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "pairwise_explanation_similarity.csv", index=False)

    freq = (
        pd.Series([roi for s in top_sets.values() for roi in s])
        .value_counts()
        .rename_axis("roi").reset_index(name="topk_count")
    )
    freq["selection_frequency"] = freq["topk_count"] / max(len(top_sets), 1)
    freq.to_csv(out / "roi_selection_frequency.csv", index=False)
    print(f"Wrote stability summaries to {out}")
    print("For the paper, repeat the entire outer CV over multiple seeds and compute stability across independently fitted models, not just across subjects.")

if __name__ == "__main__":
    main()
