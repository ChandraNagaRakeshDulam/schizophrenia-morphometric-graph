#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_utils import apply_node_scaler, make_graph
from models import GraphClassifier


@torch.no_grad()
def model_score(model, graph, device):

    graph = graph.to(device)

    batch = torch.zeros(
        graph.num_nodes,
        dtype=torch.long,
        device=device
    )

    logit = model(
        graph.x,
        graph.edge_index,
        batch
    )

    return float(
        torch.sigmoid(logit)
        .view(-1)[0]
        .cpu()
    )


def bootstrap_mean(values, n_boot, seed):

    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        boots[b] = sample.mean()

    return {
        "mean":
            float(values.mean()),

        "median":
            float(np.median(values)),

        "lower_95":
            float(np.percentile(boots, 2.5)),

        "upper_95":
            float(np.percentile(boots, 97.5)),

        "positive_fraction":
            float(np.mean(values > 0)),
    }


def stratified_bootstrap_mean(
    df,
    value_col,
    n_boot,
    seed
):

    rng = np.random.default_rng(seed)

    groups = {
        int(label):
            g[value_col].to_numpy(dtype=float)
        for label, g in df.groupby("y")
    }

    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):

        sampled = []

        for values in groups.values():

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

    values = df[value_col].to_numpy(dtype=float)

    return {
        "mean":
            float(values.mean()),

        "median":
            float(np.median(values)),

        "lower_95":
            float(np.percentile(boots, 2.5)),

        "upper_95":
            float(np.percentile(boots, 97.5)),

        "positive_fraction":
            float(np.mean(values > 0)),
    }


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True
    )

    p.add_argument(
        "--checkpoint",
        required=True
    )

    p.add_argument(
        "--node-attributions",
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

    # --------------------------------------------------
    # External dataset
    # --------------------------------------------------

    d = np.load(
        a.dataset,
        allow_pickle=True
    )

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(np.float32)
    A = d["A_msn"].astype(np.float32)

    roi_names = d[
        "roi_names"
    ].astype(str)

    feature_names = d[
        "feature_names"
    ].astype(str)

    n_rois = len(roi_names)

    if max(ks) >= n_rois:

        raise ValueError(
            f"Largest k must be smaller than {n_rois}"
        )

    roi_to_idx = {
        roi: i
        for i, roi in enumerate(roi_names)
    }

    # --------------------------------------------------
    # Frozen ds000030 model
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    c = torch.load(
        a.checkpoint,
        map_location=device,
        weights_only=False
    )

    assert c["representation"] == "msn"
    assert c["model_kind"] == "gat"
    assert float(c["density"]) == 0.20
    assert int(c["hidden"]) == 64
    assert c["rank_mode"] == "absolute"
    assert c["uses_edge_weights"] is False

    assert np.array_equal(
        roi_names,
        np.asarray(c["roi_names"]).astype(str)
    )

    assert np.array_equal(
        feature_names,
        np.asarray(c["feature_names"]).astype(str)
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

    # IMPORTANT:
    # external data are transformed with the scaler
    # learned from all ds000030 development subjects.
    Xs = apply_node_scaler(
        X,
        c["mean"],
        c["std"]
    )

    # --------------------------------------------------
    # External GNNExplainer attribution
    # --------------------------------------------------

    attr = pd.read_csv(
        a.node_attributions
    )

    required = {
        "subject_id",
        "subject_index",
        "y",
        "pred",
        "correct",
        "roi",
        "feature",
        "importance",
    }

    missing = required - set(attr.columns)

    if missing:
        raise RuntimeError(
            f"Missing attribution columns: {sorted(missing)}"
        )

    assert attr["subject_id"].nunique() == 71

    # Same aggregation as internal faithfulness analysis:
    # five feature masks -> one ROI score.
    roi_attr = (
        attr.groupby(
            [
                "subject_id",
                "subject_index",
                "y",
                "pred",
                "correct",
                "roi"
            ],
            as_index=False
        )["importance"]
        .mean()
    )

    print("Device:", device)
    print("Subjects:", roi_attr.subject_id.nunique())
    print("k values:", ks)
    print("Random repetitions:", a.random_repeats)
    print("Frozen density:", c["density"])
    print("Frozen hidden:", c["hidden"])

    rows = []

    # --------------------------------------------------
    # Subject-level faithfulness
    # --------------------------------------------------

    for position, (sid, sg) in enumerate(
        roi_attr.groupby("subject_id"),
        start=1
    ):

        idx = int(
            sg["subject_index"].iloc[0]
        )

        true_y = int(
            sg["y"].iloc[0]
        )

        # Build original external graph.
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

        # Attribution file and actual frozen-model prediction
        # should agree.
        stored_pred = int(
            sg["pred"].iloc[0]
        )

        if original_pred != stored_pred:

            raise RuntimeError(
                f"{sid}: prediction mismatch "
                f"faithfulness={original_pred}, "
                f"attribution={stored_pred}"
            )

        original_conf = (
            original_score
            if original_pred == 1
            else 1.0 - original_score
        )

        # Deterministic ranking.
        ranked_rois = (
            sg.sort_values(
                ["importance", "roi"],
                ascending=[False, True]
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

        # Nested matched random perturbations.
        local_seed = (
            a.seed
            + int(idx)
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

            # ==========================================
            # GNNExplainer-selected top-k perturbation
            # ==========================================

            selected = ranked_idx[:k]

            X_top = Xs[idx].copy()

            # Because Xs uses the frozen internal scaler,
            # zero corresponds to the ds000030 training mean.
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

            # ==========================================
            # Matched random top-k perturbations
            # ==========================================

            random_conf_drops = []
            random_sz_drops = []
            random_flips = []

            for r in range(
                a.random_repeats
            ):

                rand_idx = (
                    random_permutations[r, :k]
                )

                X_rand = Xs[idx].copy()

                X_rand[rand_idx, :] = 0.0

                rand_graph = make_graph(
                    X_rand,
                    A[idx],
                    y[idx],
                    float(c["density"]),
                    c["rank_mode"],
                    idx
                )

                rand_score = model_score(
                    model,
                    rand_graph,
                    device
                )

                rand_pred = int(
                    rand_score >= 0.5
                )

                rand_conf = (
                    rand_score
                    if original_pred == 1
                    else 1.0 - rand_score
                )

                random_conf_drops.append(
                    original_conf
                    - rand_conf
                )

                random_sz_drops.append(
                    original_score
                    - rand_score
                )

                random_flips.append(
                    int(
                        rand_pred
                        != original_pred
                    )
                )

            random_conf_drop = float(
                np.mean(
                    random_conf_drops
                )
            )

            random_sz_drop = float(
                np.mean(
                    random_sz_drops
                )
            )

            random_flip_rate = float(
                np.mean(
                    random_flips
                )
            )

            rows.append({
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

                "top_conf_drop":
                    float(
                        top_conf_drop
                    ),

                "random_conf_drop":
                    random_conf_drop,

                "confidence_advantage":
                    float(
                        top_conf_drop
                        - random_conf_drop
                    ),

                "top_sz_probability_drop":
                    float(
                        top_sz_drop
                    ),

                "random_sz_probability_drop":
                    random_sz_drop,

                "sz_probability_advantage":
                    float(
                        top_sz_drop
                        - random_sz_drop
                    ),

                "top_flip":
                    int(
                        top_flip
                    ),

                "random_flip_rate":
                    random_flip_rate,

                "flip_advantage":
                    float(
                        top_flip
                        - random_flip_rate
                    ),
            })

        if (
            position % 10 == 0
            or position == 71
        ):

            print(
                f"processed {position}/71"
            )

    df = pd.DataFrame(rows)

    out = Path(a.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        out /
        "external_multik_faithfulness.csv",
        index=False
    )

    # --------------------------------------------------
    # Subject-level bootstrap summaries
    # --------------------------------------------------

    summary_rows = []
    summary_json = {}

    for k in ks:

        q = df[
            df.k == k
        ].copy()

        assert len(q) == 71

        all_stats = stratified_bootstrap_mean(
            q,
            "confidence_advantage",
            a.bootstrap,
            a.seed + k
        )

        # All true SZ subjects.
        sz = q[
            q.y == 1
        ].copy()

        sz_stats = bootstrap_mean(
            sz["confidence_advantage"],
            a.bootstrap,
            a.seed + 1000 + k
        )

        # TP-SZ sensitivity analysis:
        # direct change in schizophrenia probability.
        tp_sz = q[
            (q.y == 1)
            & (q.original_pred == 1)
        ].copy()

        tp_stats = bootstrap_mean(
            tp_sz["sz_probability_advantage"],
            a.bootstrap,
            a.seed + 2000 + k
        )

        all_top = float(
            q.top_conf_drop.mean()
        )

        all_random = float(
            q.random_conf_drop.mean()
        )

        all_top_flip = float(
            q.top_flip.mean()
        )

        all_random_flip = float(
            q.random_flip_rate.mean()
        )

        sz_top = float(
            sz.top_conf_drop.mean()
        )

        sz_random = float(
            sz.random_conf_drop.mean()
        )

        tp_top_sz = float(
            tp_sz.top_sz_probability_drop.mean()
        )

        tp_random_sz = float(
            tp_sz.random_sz_probability_drop.mean()
        )

        tp_top_flip = float(
            tp_sz.top_flip.mean()
        )

        tp_random_flip = float(
            tp_sz.random_flip_rate.mean()
        )

        summary_rows.append({
            "group":
                "ALL",

            "k":
                k,

            "n":
                len(q),

            "top_drop":
                all_top,

            "random_drop":
                all_random,

            "advantage":
                all_stats["mean"],

            "ci_lower":
                all_stats["lower_95"],

            "ci_upper":
                all_stats["upper_95"],

            "positive_fraction":
                all_stats[
                    "positive_fraction"
                ],

            "top_flip_rate":
                all_top_flip,

            "random_flip_rate":
                all_random_flip,
        })

        summary_rows.append({
            "group":
                "SZ",

            "k":
                k,

            "n":
                len(sz),

            "top_drop":
                sz_top,

            "random_drop":
                sz_random,

            "advantage":
                sz_stats["mean"],

            "ci_lower":
                sz_stats["lower_95"],

            "ci_upper":
                sz_stats["upper_95"],

            "positive_fraction":
                sz_stats[
                    "positive_fraction"
                ],

            "top_flip_rate":
                float(
                    sz.top_flip.mean()
                ),

            "random_flip_rate":
                float(
                    sz.random_flip_rate.mean()
                ),
        })

        summary_rows.append({
            "group":
                "TP_SZ",

            "k":
                k,

            "n":
                len(tp_sz),

            "top_drop":
                tp_top_sz,

            "random_drop":
                tp_random_sz,

            "advantage":
                tp_stats["mean"],

            "ci_lower":
                tp_stats["lower_95"],

            "ci_upper":
                tp_stats["upper_95"],

            "positive_fraction":
                tp_stats[
                    "positive_fraction"
                ],

            "top_flip_rate":
                tp_top_flip,

            "random_flip_rate":
                tp_random_flip,
        })

        summary_json[str(k)] = {
            "ALL": {
                "n":
                    int(len(q)),

                "top_confidence_drop":
                    all_top,

                "random_confidence_drop":
                    all_random,

                "advantage":
                    all_stats,
            },

            "SZ": {
                "n":
                    int(len(sz)),

                "top_confidence_drop":
                    sz_top,

                "random_confidence_drop":
                    sz_random,

                "advantage":
                    sz_stats,
            },

            "TP_SZ": {
                "n":
                    int(len(tp_sz)),

                "top_sz_probability_drop":
                    tp_top_sz,

                "random_sz_probability_drop":
                    tp_random_sz,

                "advantage":
                    tp_stats,
            },
        }

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        out /
        "external_multik_faithfulness_summary.csv",
        index=False
    )

    (
        out /
        "external_multik_faithfulness_summary.json"
    ).write_text(
        json.dumps(
            summary_json,
            indent=2
        )
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print()
    print(
        "======================================"
    )
    print(
        "EXTERNAL MULTI-K ROI FAITHFULNESS"
    )
    print(
        "======================================"
    )

    for group in [
        "ALL",
        "SZ",
        "TP_SZ"
    ]:

        print()
        print(group)
        print(
            "--------------------------------------"
        )

        q = summary[
            summary.group == group
        ]

        for _, r in q.iterrows():

            print(
                f"k={int(r.k):2d} "
                f"n={int(r.n):2d} "
                f"top={r.top_drop:.4f} "
                f"random={r.random_drop:.4f} "
                f"adv={r.advantage:.4f} "
                f"95% CI "
                f"[{r.ci_lower:.4f}, "
                f"{r.ci_upper:.4f}] "
                f"positive="
                f"{r.positive_fraction:.3f}"
            )

            print(
                f"     flip: "
                f"top={r.top_flip_rate:.4f} "
                f"random={r.random_flip_rate:.4f}"
            )

    print()
    print("Output:", out)


if __name__ == "__main__":
    main()
