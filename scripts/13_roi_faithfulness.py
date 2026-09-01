#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader

from graph_utils import apply_node_scaler, make_graph
from models import GraphClassifier


@torch.no_grad()
def score_model(model, g, device):
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


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True)
    p.add_argument("--gnn-results", required=True)
    p.add_argument("--node-attributions", required=True)
    p.add_argument("--representation",
                   choices=["msn","mind"],
                   required=True)

    p.add_argument("--top-k",
                   type=int,
                   default=10)

    p.add_argument("--random-repeats",
                   type=int,
                   default=100)

    p.add_argument("--out",
                   required=True)

    p.add_argument("--seed",
                   type=int,
                   default=42)

    a = p.parse_args()

    rng = np.random.default_rng(a.seed)

    d = np.load(
        a.dataset,
        allow_pickle=True
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(np.float32)

    A = d[
        "A_msn"
        if a.representation == "msn"
        else "A_mind"
    ].astype(np.float32)

    roi_names = d["roi_names"].astype(str)

    roi_to_idx = {
        roi: i
        for i, roi in enumerate(roi_names)
    }

    attr = pd.read_csv(
        a.node_attributions
    )

    # Aggregate feature attribution -> ROI attribution
    roi_attr = (
        attr.groupby(
            [
                "split",
                "subject_id",
                "subject_index",
                "y",
                "pred",
                "correct",
                "roi"
            ],
            as_index=False
        )
        ["importance"]
        .mean()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    rows = []

    split_ids = sorted(
        roi_attr["split"].unique()
    )

    for split_id in split_ids:

        ckpt_path = (
            Path(a.gnn_results) /
            f"split_{int(split_id)}.pt"
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

        Xs = apply_node_scaler(
            X,
            c["mean"],
            c["std"]
        )

        split_attr = roi_attr[
            roi_attr["split"] == split_id
        ]

        for sid, sg in split_attr.groupby(
            "subject_id"
        ):

            idx = int(
                sg["subject_index"].iloc[0]
            )

            true_y = int(
                sg["y"].iloc[0]
            )

            pred_y = int(
                sg["pred"].iloc[0]
            )

            # Original held-out graph
            g = make_graph(
                Xs[idx],
                A[idx],
                y[idx],
                float(c["density"]),
                c["rank_mode"],
                idx
            )

            original_score = score_model(
                model,
                g,
                device
            )

            top_rois = (
                sg.sort_values(
                    "importance",
                    ascending=False
                )
                .head(a.top_k)
                ["roi"]
                .tolist()
            )

            top_idx = np.array(
                [roi_to_idx[r] for r in top_rois],
                dtype=int
            )

            # -------------------------------
            # Perturb explainer-selected ROIs
            # -------------------------------

            X_top = Xs[idx].copy()

            # 0 = training-fold mean after scaling
            X_top[top_idx, :] = 0.0

            g_top = make_graph(
                X_top,
                A[idx],
                y[idx],
                float(c["density"]),
                c["rank_mode"],
                idx
            )

            top_score = score_model(
                model,
                g_top,
                device
            )

            # Confidence in original predicted class
            original_conf = (
                original_score
                if pred_y == 1
                else 1.0 - original_score
            )

            top_conf = (
                top_score
                if pred_y == 1
                else 1.0 - top_score
            )

            top_conf_drop = (
                original_conf - top_conf
            )

            # For true SZ, also measure direct
            # schizophrenia probability drop.
            sz_score_drop = (
                original_score - top_score
            )

            # -------------------------------
            # Random matched perturbations
            # -------------------------------

            random_conf_drops = []
            random_sz_drops = []

            all_nodes = np.arange(
                len(roi_names)
            )

            for _ in range(
                a.random_repeats
            ):

                rand_idx = rng.choice(
                    all_nodes,
                    size=a.top_k,
                    replace=False
                )

                X_rand = Xs[idx].copy()

                X_rand[rand_idx, :] = 0.0

                g_rand = make_graph(
                    X_rand,
                    A[idx],
                    y[idx],
                    float(c["density"]),
                    c["rank_mode"],
                    idx
                )

                rand_score = score_model(
                    model,
                    g_rand,
                    device
                )

                rand_conf = (
                    rand_score
                    if pred_y == 1
                    else 1.0 - rand_score
                )

                random_conf_drops.append(
                    original_conf - rand_conf
                )

                random_sz_drops.append(
                    original_score - rand_score
                )

            rows.append({
                "split": int(split_id),
                "subject_id": sid,
                "subject_index": idx,
                "y": true_y,
                "pred": pred_y,
                "correct":
                    int(pred_y == true_y),

                "original_score":
                    original_score,

                "topk_score":
                    top_score,

                "topk_predicted_class_conf_drop":
                    top_conf_drop,

                "random_mean_predicted_class_conf_drop":
                    float(
                        np.mean(
                            random_conf_drops
                        )
                    ),

                "faithfulness_advantage":
                    float(
                        top_conf_drop -
                        np.mean(
                            random_conf_drops
                        )
                    ),

                "topk_sz_score_drop":
                    sz_score_drop,

                "random_mean_sz_score_drop":
                    float(
                        np.mean(
                            random_sz_drops
                        )
                    ),

                "top_rois":
                    "|".join(top_rois)
            })

    result = pd.DataFrame(rows)

    out = Path(a.out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        out /
        "roi_faithfulness.csv",
        index=False
    )

    print(
        "Instances:",
        len(result)
    )

    print(
        "\nALL HELD-OUT INSTANCES"
    )

    print(
        "Top-k predicted-class confidence drop:",
        result[
            "topk_predicted_class_conf_drop"
        ].mean()
    )

    print(
        "Random predicted-class confidence drop:",
        result[
            "random_mean_predicted_class_conf_drop"
        ].mean()
    )

    print(
        "Mean faithfulness advantage:",
        result[
            "faithfulness_advantage"
        ].mean()
    )

    tp = result[
        (result.y == 1) &
        (result.pred == 1)
    ]

    print(
        "\nTRUE-POSITIVE SCHIZOPHRENIA INSTANCES:",
        len(tp)
    )

    if len(tp):

        print(
            "Top-k SZ probability drop:",
            tp[
                "topk_sz_score_drop"
            ].mean()
        )

        print(
            "Random SZ probability drop:",
            tp[
                "random_mean_sz_score_drop"
            ].mean()
        )

        print(
            "Difference:",
            (
                tp[
                    "topk_sz_score_drop"
                ] -
                tp[
                    "random_mean_sz_score_drop"
                ]
            ).mean()
        )


if __name__ == "__main__":
    main()
