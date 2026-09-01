#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_geometric.explain import Explainer, GNNExplainer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_utils import apply_node_scaler, make_graph
from models import GraphClassifier


def checkpoint_number(path):
    return int(path.stem.split("_")[-1])


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True)
    p.add_argument("--gnn-results", required=True)
    p.add_argument("--representation",
                   choices=["msn", "mind"],
                   required=True)
    p.add_argument("--out", required=True)

    # Useful for smoke testing.
    p.add_argument("--max-splits", type=int, default=None)
    p.add_argument("--max-subjects", type=int, default=None)

    p.add_argument("--explainer-epochs",
                   type=int,
                   default=200)

    p.add_argument("--seed",
                   type=int,
                   default=12345)

    a = p.parse_args()

    d = np.load(a.dataset, allow_pickle=True)

    ids = d["ids"].astype(str)
    y = d["y"].astype(int)

    X = d["X_nodes"].astype(np.float32)

    A = d[
        "A_mind"
        if a.representation == "mind"
        else "A_msn"
    ].astype(np.float32)

    roi_names = d["roi_names"].astype(str)
    feat_names = d["feature_names"].astype(str)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Final GNN pipeline uses split_0.pt ... split_14.pt
    ckpts = sorted(
        Path(a.gnn_results).glob("split_*.pt"),
        key=checkpoint_number
    )

    # Backwards-compatible fallback.
    if not ckpts:
        ckpts = sorted(
            Path(a.gnn_results).glob("fold_*.pt"),
            key=checkpoint_number
        )

    if not ckpts:
        raise RuntimeError(
            f"No checkpoints found in {a.gnn_results}"
        )

    if a.max_splits is not None:
        ckpts = ckpts[:a.max_splits]

    print("Device:", device)
    print("Checkpoints:", len(ckpts))

    roi_rows = []
    edge_rows = []

    explained_subjects = 0

    for ckpt_path in ckpts:

        c = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False
        )

        parsed_id = checkpoint_number(ckpt_path)

        split_id = int(
            c.get("split", parsed_id)
        )

        repeat = int(
            c.get("repeat", split_id // 5)
        )

        fold = int(
            c.get("fold", split_id % 5)
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

        # Scaling parameters came only from the corresponding
        # outer-training fold.
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
                return_type="raw",
            ),
        )

        test_indices = np.asarray(
            c["test_indices"],
            dtype=int
        )

        if a.max_subjects is not None:
            test_indices = test_indices[:a.max_subjects]

        print(
            f"split={split_id} "
            f"repeat={repeat} "
            f"fold={fold} "
            f"subjects={len(test_indices)} "
            f"density={c['density']} "
            f"hidden={c['hidden']}"
        )

        for idx in test_indices:

            # Deterministic explainer initialization.
            explanation_seed = (
                a.seed +
                split_id * 10000 +
                int(idx)
            )

            torch.manual_seed(explanation_seed)

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
                idx
            ).to(device)

            batch = torch.zeros(
                g.num_nodes,
                dtype=torch.long,
                device=device
            )

            # Save original model prediction.
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

            pred = int(score >= 0.5)

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

            # ROI × feature attribution.
            for r in range(node_mask.shape[0]):

                for f in range(node_mask.shape[1]):

                    roi_rows.append({
                        "split": split_id,
                        "repeat": repeat,
                        "fold": fold,
                        "subject_id": ids[idx],
                        "subject_index": int(idx),
                        "y": int(y[idx]),
                        "score": score,
                        "pred": pred,
                        "correct":
                            int(pred == int(y[idx])),
                        "density":
                            float(c["density"]),
                        "hidden":
                            int(c["hidden"]),
                        "roi":
                            roi_names[r],
                        "feature":
                            feat_names[f],
                        "importance":
                            float(node_mask[r, f]),
                        "explainer_seed":
                            explanation_seed,
                    })

            # Directed edge entries.
            # We aggregate the two directions later.
            for e, imp in enumerate(edge_mask):

                u = int(edge_index[0, e])
                v = int(edge_index[1, e])

                edge_rows.append({
                    "split": split_id,
                    "repeat": repeat,
                    "fold": fold,
                    "subject_id": ids[idx],
                    "subject_index": int(idx),
                    "y": int(y[idx]),
                    "score": score,
                    "pred": pred,
                    "correct":
                        int(pred == int(y[idx])),
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

            explained_subjects += 1

    roi_df = pd.DataFrame(roi_rows)
    edge_df = pd.DataFrame(edge_rows)

    if roi_df.empty or edge_df.empty:
        raise RuntimeError(
            "No explanations were generated."
        )

    roi_df.to_csv(
        out / "node_feature_attributions.csv",
        index=False
    )

    edge_df.to_csv(
        out / "edge_attributions.csv",
        index=False
    )

    print()
    print("Explanation generation complete.")
    print("Explained subject-fold instances:",
          explained_subjects)
    print("Node-feature attribution rows:",
          len(roi_df))
    print("Edge attribution rows:",
          len(edge_df))
    print("Output:", out)


if __name__ == "__main__":
    main()
