#!/usr/bin/env python3

import json
from pathlib import Path
import pandas as pd

OUT = Path("results/paper")
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text())


# -------------------------------------------------------
# External classification
# -------------------------------------------------------

models = {
    "ROI-RF":
        "results/external/ds004302_roi_rf/external_metrics.json",

    "MSN-GAT":
        "results/external/ds004302_msn_gat/external_metrics.json",

    "MIND-GAT":
        "results/external/ds004302_mind_gat/external_metrics.json",
}

rows = []

for name, path in models.items():

    d = load_json(path)

    m = d["metrics"]
    ci = d["bootstrap_95_ci"]

    rows.append({
        "Model": name,

        "ROC-AUC":
            m["roc_auc"],

        "ROC-AUC CI lower":
            ci["roc_auc"]["lower_95"],

        "ROC-AUC CI upper":
            ci["roc_auc"]["upper_95"],

        "PR-AUC":
            m["pr_auc"],

        "PR-AUC CI lower":
            ci["pr_auc"]["lower_95"],

        "PR-AUC CI upper":
            ci["pr_auc"]["upper_95"],

        "Balanced accuracy":
            m["balanced_accuracy"],

        "Sensitivity":
            m["sensitivity"],

        "Specificity":
            m["specificity"],

        "Precision":
            m["precision"],

        "F1":
            m["f1"],

        "TN": m["tn"],
        "FP": m["fp"],
        "FN": m["fn"],
        "TP": m["tp"],
    })

external = pd.DataFrame(rows)

external.to_csv(
    OUT / "table_external_classification.csv",
    index=False
)


# -------------------------------------------------------
# Paired model comparisons
# -------------------------------------------------------

three = load_json(
    "results/external/three_model_comparison/"
    "three_model_comparison.json"
)

pair_rows = []

for comparison, metrics in (
    three["paired_bootstrap"].items()
):

    for metric, x in metrics.items():

        pair_rows.append({
            "Comparison":
                comparison,

            "Metric":
                metric,

            "Difference":
                x["observed"],

            "95% CI lower":
                x["lower_95"],

            "95% CI upper":
                x["upper_95"],

            "P(delta > 0)":
                x["prob_delta_gt_0"],
        })

pairs = pd.DataFrame(pair_rows)

pairs.to_csv(
    OUT / "table_external_paired_comparisons.csv",
    index=False
)


# -------------------------------------------------------
# Cross-cohort explanation reproducibility
# -------------------------------------------------------

repro = load_json(
    "results/explain_cross_cohort/msn_gat/"
    "cross_cohort_reproducibility.json"
)

repro_rows = []

for group in ["CONTROL", "SZ"]:

    r = repro[group]

    repro_rows.append({
        "Group": group,

        "Internal N":
            r["n_internal"],

        "External N":
            r["n_external"],

        "ROI Spearman":
            r["roi_spearman"],

        "Top-10 Jaccard":
            r["top_10_jaccard"],

        "Top-10 overlap":
            "; ".join(
                r["top_10_overlap"]
            ),

        "Bootstrap Spearman CI lower":
            r["bootstrap"]["spearman"]["lower_95"],

        "Bootstrap Spearman CI upper":
            r["bootstrap"]["spearman"]["upper_95"],

        "Bootstrap Jaccard CI lower":
            r["bootstrap"]["topk_jaccard"]["lower_95"],

        "Bootstrap Jaccard CI upper":
            r["bootstrap"]["topk_jaccard"]["upper_95"],
    })

repro_df = pd.DataFrame(repro_rows)

repro_df.to_csv(
    OUT / "table_cross_cohort_reproducibility.csv",
    index=False
)


# -------------------------------------------------------
# Feature reproducibility
# -------------------------------------------------------

feature_rows = []

for group in ["CONTROL", "SZ"]:

    r = repro[
        "feature_reproducibility"
    ][group]

    internal = {
        x["feature"]: x["importance"]
        for x in r["internal_ranking"]
    }

    external_f = {
        x["feature"]: x["importance"]
        for x in r["external_ranking"]
    }

    for feature in [
        "CT",
        "MC",
        "SD",
        "SA",
        "Vol",
    ]:

        feature_rows.append({
            "Group":
                group,

            "Feature":
                feature,

            "Internal importance":
                internal[feature],

            "External importance":
                external_f[feature],

            "Feature Spearman":
                r["spearman"],
        })

feature_df = pd.DataFrame(
    feature_rows
)

feature_df.to_csv(
    OUT / "table_feature_reproducibility.csv",
    index=False
)


# -------------------------------------------------------
# External faithfulness
# -------------------------------------------------------

faith = pd.read_csv(
    "results/explain_external/"
    "msn_gat_faithfulness/"
    "external_multik_faithfulness_summary.csv"
)

faith.to_csv(
    OUT / "table_external_faithfulness.csv",
    index=False
)


# -------------------------------------------------------
# Compact headline table
# -------------------------------------------------------

headline = pd.DataFrame([
    {
        "Finding":
            "ROI-RF external ROC-AUC",
        "Value":
            "0.6965 [0.5696, 0.8157]",
    },
    {
        "Finding":
            "MSN-GAT external ROC-AUC",
        "Value":
            "0.6504 [0.5165, 0.7730]",
    },
    {
        "Finding":
            "MIND-GAT external ROC-AUC",
        "Value":
            "0.4600 [0.3226, 0.5991]",
    },
    {
        "Finding":
            "ROI-RF minus MSN-GAT ROC-AUC",
        "Value":
            "+0.0461 [-0.0783, 0.1696]",
    },
    {
        "Finding":
            "MSN-GAT minus MIND-GAT ROC-AUC",
        "Value":
            "+0.1904 [0.0670, 0.3130]",
    },
    {
        "Finding":
            "SZ cross-cohort ROI Spearman",
        "Value":
            "0.2747",
    },
    {
        "Finding":
            "SZ top-10 ROI Jaccard",
        "Value":
            "0.1111",
    },
    {
        "Finding":
            "SZ feature Spearman",
        "Value":
            "0.8000",
    },
    {
        "Finding":
            "External all-subject top-10 faithfulness advantage",
        "Value":
            "0.3486 [0.2839, 0.4150]",
    },
    {
        "Finding":
            "External SZ top-10 faithfulness advantage",
        "Value":
            "0.3129 [0.2303, 0.4002]",
    },
])

headline.to_csv(
    OUT / "headline_results.csv",
    index=False
)


print("\n======================================")
print("PAPER RESULTS TABLES CREATED")
print("======================================")

for p in sorted(OUT.glob("*.csv")):
    print(p)

print("\nExternal classification:")
print(
    external[
        [
            "Model",
            "ROC-AUC",
            "PR-AUC",
            "Balanced accuracy",
            "Sensitivity",
            "Specificity",
        ]
    ].to_string(index=False)
)

print("\nHeadline results:")
print(headline.to_string(index=False))
