#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_utils import apply_node_scaler, make_graph
from models import GraphClassifier


@torch.no_grad()
def model_score(model, g, device):
    g = g.to(device)

    batch = torch.zeros(
        g.num_nodes,
        dtype=torch.long,
        device=device
    )

    logit = model(
        g.x,
        g.edge_index,
        batch
    )

    return float(
        torch.sigmoid(logit)
        .view(-1)[0]
        .cpu()
    )


def bootstrap_ci(values, n_boot=10000, seed=42):

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
        float(values.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5))
    )


def stratified_bootstrap_ci(
    df,
    value_col,
    n_boot=10000,
    seed=42
):

    rng = np.random.default_rng(seed)

    groups = {
        label: g[value_col].to_numpy()
        for label, g in df.groupby("y")
    }

    boots = np.empty(n_boot)

    for b in range(n_boot):

        sampled = []

        for label, values in groups.items():

            sampled.append(
                rng.choice(
                    values,
                    size=len(values),
                    replace=True
                )
            )

        boots[b] = np.concatenate(
            sampled
        ).mean()

    return (
        float(df[value_col].mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5))
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True
    )

    p.add_argument(
        "--gnn-results",
        required=True
    )

    p.add_argument(
        "--node-attributions",
        required=True
    )

    p.add_argument(
        "--representation",
        choices=["msn", "mind"],
        required=True
    )

    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 5, 10, 15, 20]
    )

    p.add_argument(
        "--random-repeats",
        type=int,
        default=100
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--out",
        required=True
    )

    a = p.parse_args()

    ks = sorted(set(a.ks))

    d = np.load(
        a.dataset,
        allow_pickle=True
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(
        np.float32
    )

    A = d[
        "A_msn"
        if a.representation == "msn"
        else "A_mind"
    ].astype(np.float32)

    roi_names = d[
        "roi_names"
    ].astype(str)

    n_rois = len(roi_names)

    if max(ks) >= n_rois:
        raise ValueError(
            "Largest k must be smaller "
            f"than number of ROIs ({n_rois})."
        )

    roi_to_idx = {
        roi: i
        for i, roi in enumerate(
            roi_names
        )
    }

    attr = pd.read_csv(
        a.node_attributions
    )

    # Collapse five feature masks into
    # one ROI importance score.
    roi_attr = (
        attr.groupby(
            [
                "split",
                "repeat",
                "fold",
                "subject_id",
                "subject_index",
                "y",
                "roi"
            ],
            as_index=False
        )["importance"]
        .mean()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("k values:", ks)
    print(
        "Random repetitions:",
        a.random_repeats
    )

    rows = []

    split_ids = sorted(
        roi_attr["split"].unique()
    )

    for split_id in split_ids:

        ckpt_path = (
            Path(a.gnn_results)
            / f"split_{int(split_id)}.pt"
        )

        c = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False
        )

        model = GraphClassifier(
            X.shape[-1],
            int(c["hidden"]),
            c["model_kind"]
        ).to(device)

        model.load_state_dict(
            c["state_dict"]
        )

        model.eval()

        # Fold-specific training-only scaler.
        Xs = apply_node_scaler(
            X,
            c["mean"],
            c["std"]
        )

        split_attr = roi_attr[
            roi_attr["split"]
            == split_id
        ]

        print(
            f"split={split_id} "
            f"subjects="
            f"{split_attr.subject_id.nunique()}"
        )

        for sid, sg in split_attr.groupby(
            "subject_id"
        ):

            idx = int(
                sg["subject_index"].iloc[0]
            )

            true_y = int(
                sg["y"].iloc[0]
            )

            # Original held-out graph.
            original_graph = make_graph(
                Xs[idx],
                A[idx],
                y[idx],
                float(c["density"]),
                c["rank_mode"],
                idx
            )

            original_score = model_score(
                model,
                original_graph,
                device
            )

            original_pred = int(
                original_score >= 0.5
            )

            original_conf = (
                original_score
                if original_pred == 1
                else 1.0 - original_score
            )

            # Stable ROI ranking for this
            # subject/model explanation.
            ranked_rois = (
                sg.sort_values(
                    [
                        "importance",
                        "roi"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                )["roi"]
                .tolist()
            )

            ranked_idx = np.asarray(
                [
                    roi_to_idx[r]
                    for r in ranked_rois
                ],
                dtype=int
            )

            # Deterministic random permutations.
            # Same permutations are reused across
            # all k values, giving nested controls.
            local_seed = (
                a.seed
                + int(split_id) * 100000
                + idx
            )

            rng = np.random.default_rng(
                local_seed
            )

            random_permutations = np.stack(
                [
                    rng.permutation(n_rois)
                    for _ in range(
                        a.random_repeats
                    )
                ]
            )

            for k in ks:

                # ---------------------------------
                # GNNExplainer-selected ROI masking
                # ---------------------------------

                selected = ranked_idx[:k]

                X_top = Xs[idx].copy()

                # Standardized 0 corresponds
                # approximately to training mean.
                X_top[selected, :] = 0.0

                top_graph = make_graph(
                    X_top,
                    A[idx],
                    y[idx],
                    float(c["density"]),
                    c["rank_mode"],
                    idx
                )

                top_score = model_score(
                    model,
                    top_graph,
                    device
                )

                top_pred = int(
                    top_score >= 0.5
                )

                top_conf = (
                    top_score
                    if original_pred == 1
                    else 1.0 - top_score
                )

                top_conf_drop = (
                    original_conf
                    - top_conf
                )

                top_sz_drop = (
                    original_score
                    - top_score
                )

                top_flip = int(
                    top_pred != original_pred
                )

                # ---------------------------------
                # Matched random ROI masking
                # ---------------------------------

                random_conf_drops = []
                random_sz_drops = []
                random_flips = []

                for permutation in (
                    random_permutations
                ):

                    random_idx = (
                        permutation[:k]
                    )

                    X_rand = Xs[idx].copy()

                    X_rand[
                        random_idx, :
                    ] = 0.0

                    random_graph = make_graph(
                        X_rand,
                        A[idx],
                        y[idx],
                        float(c["density"]),
                        c["rank_mode"],
                        idx
                    )

                    random_score = model_score(
                        model,
                        random_graph,
                        device
                    )

                    random_pred = int(
                        random_score >= 0.5
                    )

                    random_conf = (
                        random_score
                        if original_pred == 1
                        else 1.0 - random_score
                    )

                    random_conf_drops.append(
                        original_conf
                        - random_conf
                    )

                    random_sz_drops.append(
                        original_score
                        - random_score
                    )

                    random_flips.append(
                        int(
                            random_pred
                            != original_pred
                        )
                    )

                random_mean_conf_drop = (
                    float(
                        np.mean(
                            random_conf_drops
                        )
                    )
                )

                random_mean_sz_drop = (
                    float(
                        np.mean(
                            random_sz_drops
                        )
                    )
                )

                rows.append({
                    "split":
                        int(split_id),
                    "repeat":
                        int(
                            sg["repeat"].iloc[0]
                        ),
                    "fold":
                        int(
                            sg["fold"].iloc[0]
                        ),
                    "subject_id":
                        sid,
                    "subject_index":
                        idx,
                    "y":
                        true_y,
                    "original_score":
                        original_score,
                    "original_pred":
                        original_pred,
                    "correct":
                        int(
                            original_pred
                            == true_y
                        ),
                    "k":
                        int(k),

                    "top_score":
                        top_score,

                    "top_predicted_class_conf_drop":
                        top_conf_drop,

                    "random_mean_predicted_class_conf_drop":
                        random_mean_conf_drop,

                    "faithfulness_advantage":
                        top_conf_drop
                        - random_mean_conf_drop,

                    "top_sz_score_drop":
                        top_sz_drop,

                    "random_mean_sz_score_drop":
                        random_mean_sz_drop,

                    "sz_faithfulness_advantage":
                        top_sz_drop
                        - random_mean_sz_drop,

                    "top_prediction_flip":
                        top_flip,

                    "random_prediction_flip_rate":
                        float(
                            np.mean(
                                random_flips
                            )
                        ),

                    "top_rois":
                        "|".join(
                            ranked_rois[:k]
                        )
                })

    result = pd.DataFrame(rows)

    out = Path(a.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        out /
        "multik_roi_faithfulness_instances.csv",
        index=False
    )

    print("\nInstance rows:", len(result))

    # =====================================================
    # SUBJECT-LEVEL ANALYSIS
    # Average repeated-CV instances before inference.
    # =====================================================

    subject = (
        result.groupby(
            [
                "subject_id",
                "y",
                "k"
            ],
            as_index=False
        )
        .agg(
            top_drop=(
                "top_predicted_class_conf_drop",
                "mean"
            ),
            random_drop=(
                "random_mean_predicted_class_conf_drop",
                "mean"
            ),
            advantage=(
                "faithfulness_advantage",
                "mean"
            ),
            top_flip_rate=(
                "top_prediction_flip",
                "mean"
            ),
            random_flip_rate=(
                "random_prediction_flip_rate",
                "mean"
            ),
            n_instances=(
                "split",
                "size"
            )
        )
    )

    subject.to_csv(
        out /
        "multik_roi_faithfulness_subjects.csv",
        index=False
    )

    summary_rows = []

    # ---------------------------
    # All independent subjects
    # ---------------------------

    for k in ks:

        g = subject[
            subject.k == k
        ]

        mean, lo, hi = (
            stratified_bootstrap_ci(
                g,
                "advantage",
                n_boot=a.bootstrap,
                seed=a.seed + k
            )
        )

        summary_rows.append({
            "group": "ALL",
            "k": k,
            "n_subjects": len(g),
            "top_mean_drop":
                g.top_drop.mean(),
            "random_mean_drop":
                g.random_drop.mean(),
            "mean_advantage":
                mean,
            "ci_low":
                lo,
            "ci_high":
                hi,
            "positive_advantage_fraction":
                (g.advantage > 0).mean(),
            "top_flip_rate":
                g.top_flip_rate.mean(),
            "random_flip_rate":
                g.random_flip_rate.mean()
        })

    # ---------------------------
    # True SZ subjects
    # ---------------------------

    for k in ks:

        g = subject[
            (subject.k == k)
            & (subject.y == 1)
        ]

        mean, lo, hi = bootstrap_ci(
            g.advantage,
            n_boot=a.bootstrap,
            seed=a.seed + 1000 + k
        )

        summary_rows.append({
            "group": "SZ",
            "k": k,
            "n_subjects": len(g),
            "top_mean_drop":
                g.top_drop.mean(),
            "random_mean_drop":
                g.random_drop.mean(),
            "mean_advantage":
                mean,
            "ci_low":
                lo,
            "ci_high":
                hi,
            "positive_advantage_fraction":
                (g.advantage > 0).mean(),
            "top_flip_rate":
                g.top_flip_rate.mean(),
            "random_flip_rate":
                g.random_flip_rate.mean()
        })

    # ---------------------------
    # True-positive SZ subjects
    # ---------------------------

    tp = result[
        (result.y == 1)
        & (result.original_pred == 1)
    ].copy()

    tp_subject = (
        tp.groupby(
            [
                "subject_id",
                "y",
                "k"
            ],
            as_index=False
        )
        .agg(
            top_drop=(
                "top_sz_score_drop",
                "mean"
            ),
            random_drop=(
                "random_mean_sz_score_drop",
                "mean"
            ),
            advantage=(
                "sz_faithfulness_advantage",
                "mean"
            ),
            top_flip_rate=(
                "top_prediction_flip",
                "mean"
            ),
            random_flip_rate=(
                "random_prediction_flip_rate",
                "mean"
            ),
            n_tp_instances=(
                "split",
                "size"
            )
        )
    )

    tp_subject.to_csv(
        out /
        "multik_roi_faithfulness_tp_subjects.csv",
        index=False
    )

    for k in ks:

        g = tp_subject[
            tp_subject.k == k
        ]

        mean, lo, hi = bootstrap_ci(
            g.advantage,
            n_boot=a.bootstrap,
            seed=a.seed + 2000 + k
        )

        summary_rows.append({
            "group": "TP_SZ",
            "k": k,
            "n_subjects": len(g),
            "top_mean_drop":
                g.top_drop.mean(),
            "random_mean_drop":
                g.random_drop.mean(),
            "mean_advantage":
                mean,
            "ci_low":
                lo,
            "ci_high":
                hi,
            "positive_advantage_fraction":
                (g.advantage > 0).mean(),
            "top_flip_rate":
                g.top_flip_rate.mean(),
            "random_flip_rate":
                g.random_flip_rate.mean()
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        out /
        "multik_roi_faithfulness_summary.csv",
        index=False
    )

    print(
        "\n============================================"
    )
    print(
        "MULTI-K ROI FAITHFULNESS — SUBJECT LEVEL"
    )
    print(
        "============================================"
    )

    for group in [
        "ALL",
        "SZ",
        "TP_SZ"
    ]:

        print(
            f"\n--- {group} ---"
        )

        print(
            summary[
                summary.group == group
            ][
                [
                    "k",
                    "n_subjects",
                    "top_mean_drop",
                    "random_mean_drop",
                    "mean_advantage",
                    "ci_low",
                    "ci_high",
                    "positive_advantage_fraction",
                    "top_flip_rate",
                    "random_flip_rate"
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nWrote outputs to:",
        out
    )


if __name__ == "__main__":
    main()
