#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_geometric.explain import Explainer, GNNExplainer

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from graph_utils import apply_node_scaler, make_graph
from models import GraphClassifier


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)

    p.add_argument(
        "--explainer-epochs",
        type=int,
        default=200
    )

    p.add_argument(
        "--seed",
        type=int,
        default=12345
    )

    p.add_argument(
        "--max-subjects",
        type=int,
        default=None
    )

    a = p.parse_args()

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

    feat_names = d[
        "feature_names"
    ].astype(str)

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

    assert c["model_kind"] == "gat"
    assert c["representation"] == "msn"

    assert float(c["density"]) == 0.20
    assert int(c["hidden"]) == 64

    assert c["rank_mode"] == "absolute"
    assert c["uses_edge_weights"] is False

    # Representation compatibility.
    assert np.array_equal(
        roi_names,
        np.asarray(
            c["roi_names"]
        ).astype(str)
    )

    assert np.array_equal(
        feat_names,
        np.asarray(
            c["feature_names"]
        ).astype(str)
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
    # use INTERNAL ds000030 scaler.
    Xs = apply_node_scaler(
        X,
        c["mean"],
        c["std"]
    )

    explainer = Explainer(
        model=model,

        algorithm=GNNExplainer(
            epochs=a.explainer_epochs
        ),

        explanation_type="model",

        node_mask_type="attributes",
        edge_mask_type="object",

        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw"
        ),
    )

    out = Path(a.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    indices = np.arange(
        len(y),
        dtype=int
    )

    if a.max_subjects is not None:
        indices = indices[
            :a.max_subjects
        ]

    roi_rows = []
    edge_rows = []

    print("Device:", device)
    print("Subjects to explain:", len(indices))
    print("Density:", c["density"])
    print("Hidden:", c["hidden"])

    for position, idx in enumerate(indices, start=1):

        explanation_seed = (
            a.seed + int(idx)
        )

        torch.manual_seed(
            explanation_seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                explanation_seed
            )

        g = make_graph(
            Xs[idx],
            A[idx],
            y[idx],
            float(c["density"]),
            c["rank_mode"],
            int(idx)
        ).to(device)

        batch = torch.zeros(
            g.num_nodes,
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            logit = model(
                g.x,
                g.edge_index,
                batch
            )

            score = float(
                torch.sigmoid(logit)
                .view(-1)[0]
                .cpu()
            )

        pred = int(
            score >= 0.5
        )

        exp = explainer(
            g.x,
            g.edge_index,
            batch=batch
        )

        node_mask = (
            exp.node_mask
            .detach()
            .cpu()
            .numpy()
        )

        edge_mask = (
            exp.edge_mask
            .detach()
            .cpu()
            .numpy()
        )

        edge_index = (
            g.edge_index
            .detach()
            .cpu()
            .numpy()
        )

        edge_attr = (
            g.edge_attr
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        for r in range(
            node_mask.shape[0]
        ):

            for f in range(
                node_mask.shape[1]
            ):

                roi_rows.append({
                    "subject_id":
                        ids[idx],

                    "subject_index":
                        int(idx),

                    "y":
                        int(y[idx]),

                    "score":
                        score,

                    "pred":
                        pred,

                    "correct":
                        int(
                            pred == int(y[idx])
                        ),

                    "density":
                        float(c["density"]),

                    "hidden":
                        int(c["hidden"]),

                    "roi":
                        roi_names[r],

                    "feature":
                        feat_names[f],

                    "importance":
                        float(
                            node_mask[r, f]
                        ),

                    "explainer_seed":
                        explanation_seed,
                })

        for e, imp in enumerate(
            edge_mask
        ):

            u = int(
                edge_index[0, e]
            )

            v = int(
                edge_index[1, e]
            )

            edge_rows.append({
                "subject_id":
                    ids[idx],

                "subject_index":
                    int(idx),

                "y":
                    int(y[idx]),

                "score":
                    score,

                "pred":
                    pred,

                "correct":
                    int(
                        pred == int(y[idx])
                    ),

                "density":
                    float(c["density"]),

                "hidden":
                    int(c["hidden"]),

                "source":
                    roi_names[u],

                "target":
                    roi_names[v],

                "original_edge_weight":
                    float(edge_attr[e]),

                "importance":
                    float(imp),

                "explainer_seed":
                    explanation_seed,
            })

        if (
            position % 10 == 0
            or position == len(indices)
        ):

            print(
                f"explained "
                f"{position}/{len(indices)}"
            )

    roi_df = pd.DataFrame(
        roi_rows
    )

    edge_df = pd.DataFrame(
        edge_rows
    )

    roi_df.to_csv(
        out /
        "node_feature_attributions.csv",
        index=False
    )

    edge_df.to_csv(
        out /
        "edge_attributions.csv",
        index=False
    )

    print("\nExternal explanation generation complete.")

    print(
        "Subjects:",
        len(indices)
    )

    print(
        "Node-feature rows:",
        len(roi_df)
    )

    print(
        "Edge rows:",
        len(edge_df)
    )

    print(
        "Output:",
        out
    )


if __name__ == "__main__":
    main()
