#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import apply_node_scaler, make_graph
from models import GraphClassifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gnn-results", required=True)
    p.add_argument("--representation", choices=["msn","mind"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    ids, y = d["ids"].astype(str), d["y"].astype(int)
    X = d["X_nodes"].astype(np.float32)
    A = d["A_mind" if a.representation == "mind" else "A_msn"].astype(np.float32)
    roi_names = d["roi_names"].astype(str)
    feat_names = d["feature_names"].astype(str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    roi_rows, edge_rows = [], []

    for ckpt_path in sorted(Path(a.gnn_results).glob("fold_*.pt")):
        fold = int(ckpt_path.stem.split("_")[-1])
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = GraphClassifier(X.shape[-1], int(c["hidden"]), c["model_kind"]).to(device)
        model.load_state_dict(c["state_dict"])
        model.eval()

        Xs = apply_node_scaler(X, c["mean"], c["std"])

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

        for idx in np.asarray(c["test_indices"], dtype=int):
            g = make_graph(Xs[idx], A[idx], y[idx], float(c["density"]), c["rank_mode"], idx).to(device)
            batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
            exp = explainer(g.x, g.edge_index, batch=batch)

            node_mask = exp.node_mask.detach().cpu().numpy()
            edge_mask = exp.edge_mask.detach().cpu().numpy()
            edge_index = g.edge_index.detach().cpu().numpy()

            for r in range(node_mask.shape[0]):
                for f in range(node_mask.shape[1]):
                    roi_rows.append({
                        "fold": fold, "subject_id": ids[idx], "y": y[idx],
                        "roi": roi_names[r], "feature": feat_names[f],
                        "importance": float(node_mask[r, f]),
                    })

            for e, imp in enumerate(edge_mask):
                u, v = int(edge_index[0,e]), int(edge_index[1,e])
                edge_rows.append({
                    "fold": fold, "subject_id": ids[idx], "y": y[idx],
                    "source": roi_names[u], "target": roi_names[v],
                    "importance": float(imp),
                })

    pd.DataFrame(roi_rows).to_csv(out / "node_feature_attributions.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "edge_attributions.csv", index=False)
    print(f"Wrote explanations to {out}")
    print("Next: aggregate directed duplicate edges, compute top-k selection frequency, Jaccard, rank correlations, and perturbation faithfulness.")

if __name__ == "__main__":
    main()
