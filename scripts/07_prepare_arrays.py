#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

def canon(name):
    s = str(name).strip().lower()
    s = s.replace("left-", "lh_").replace("right-", "rh_")
    s = s.replace("left_", "lh_").replace("right_", "rh_")
    s = s.replace("lh-", "lh_").replace("rh-", "rh_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def load_and_align(path, target_names):
    d = np.load(path, allow_pickle=True)
    A = d["A"]
    names = [canon(x) for x in d["roi_names"].astype(str)]
    target = [canon(x) for x in target_names]
    if set(names) != set(target):
        missing = sorted(set(target) - set(names))
        extra = sorted(set(names) - set(target))
        raise RuntimeError(
            f"ROI-name mismatch for {path.name}. Missing={missing[:10]} Extra={extra[:10]}. "
            "Do not force alignment until naming is understood."
        )
    idx = [names.index(x) for x in target]
    return A[np.ix_(idx, idx)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--roi-dir", required=True)
    p.add_argument("--msn-dir", required=True)
    p.add_argument("--mind-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cohort = pd.read_csv(a.cohort)
    cohort = cohort.loc[cohort["include_pre_qc"].astype(str).str.lower().isin(["true", "1"])].copy()

    ids, ys, Xs, As_msn, As_mind = [], [], [], [], []
    canonical_roi_names = None
    feature_names = None

    for _, row in cohort.iterrows():
        fsid = row["fs_id"]
        roi_file = Path(a.roi_dir) / f"{fsid}.npz"
        msn_file = Path(a.msn_dir) / f"{fsid}.npz"
        mind_file = Path(a.mind_dir) / f"{fsid}.npz"
        if not (roi_file.exists() and msn_file.exists() and mind_file.exists()):
            continue

        r = np.load(roi_file, allow_pickle=True)
        X = r["X"].astype(np.float32)
        roi_names = r["roi_names"].astype(str)
        if canonical_roi_names is None:
            canonical_roi_names = roi_names
            feature_names = r["feature_names"].astype(str)
        elif not np.array_equal(roi_names, canonical_roi_names):
            raise RuntimeError(f"ROI feature ordering differs for {fsid}")

        A_msn = load_and_align(msn_file, canonical_roi_names)
        A_mind = load_and_align(mind_file, canonical_roi_names)

        ids.append(row["subject_id"])
        ys.append(int(row["diagnosis_binary"]))
        Xs.append(X)
        As_msn.append(A_msn)
        As_mind.append(A_mind)

    if not ids:
        raise SystemExit("No subjects have complete ROI/MSN/MIND outputs.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ids=np.asarray(ids, dtype=str),
        y=np.asarray(ys, dtype=np.int64),
        X_nodes=np.stack(Xs),
        A_msn=np.stack(As_msn).astype(np.float32),
        A_mind=np.stack(As_mind).astype(np.float32),
        roi_names=np.asarray(canonical_roi_names, dtype=str),
        feature_names=np.asarray(feature_names, dtype=str),
    )
    print(f"Wrote {out}: N={len(ids)}, cases={sum(ys)}, controls={len(ys)-sum(ys)}")

if __name__ == "__main__":
    main()
