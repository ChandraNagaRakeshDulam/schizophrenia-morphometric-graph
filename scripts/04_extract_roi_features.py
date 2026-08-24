#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot, read_morph_data

FEATURES = ["CT", "MC", "Vol", "SD", "SA"]
DROP_NAMES = {"unknown", "corpuscallosum", "medialwall"}

def decode_name(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)

def extract_hemi(subject_dir, hemi):
    annot_path = subject_dir / "label" / f"{hemi}.aparc.annot"
    labels, _, names = read_annot(str(annot_path), orig_ids=False)
    names = [decode_name(n) for n in names]

    arrays = {
        "CT": read_morph_data(str(subject_dir / "surf" / f"{hemi}.thickness")),
        "MC": read_morph_data(str(subject_dir / "surf" / f"{hemi}.curv")),
        "Vol": read_morph_data(str(subject_dir / "surf" / f"{hemi}.volume")),
        "SD": read_morph_data(str(subject_dir / "surf" / f"{hemi}.sulc")),
        "SA": read_morph_data(str(subject_dir / "surf" / f"{hemi}.area")),
    }

    n = len(labels)
    for k, v in arrays.items():
        if len(v) != n:
            raise ValueError(f"{subject_dir.name} {hemi} {k}: {len(v)} vertices != annotation {n}")

    rows = []
    for idx, name in enumerate(names):
        canon = name.lower().replace("-", "").replace("_", "")
        if canon in DROP_NAMES:
            continue
        mask = labels == idx
        if mask.sum() == 0:
            continue
        rows.append({
            "roi": f"{hemi}_{name}",
            "CT": float(np.mean(arrays["CT"][mask])),
            "MC": float(np.mean(arrays["MC"][mask])),
            "Vol": float(np.sum(arrays["Vol"][mask])),
            "SD": float(np.mean(arrays["SD"][mask])),
            "SA": float(np.sum(arrays["SA"][mask])),
            "n_vertices": int(mask.sum()),
        })
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_long = []

    for sd in sorted(Path(a.subjects_dir).glob(f"{a.dataset}_sub-*")):
        rows = extract_hemi(sd, "lh") + extract_hemi(sd, "rh")
        df = pd.DataFrame(rows)

        if len(df) != 68:
            raise RuntimeError(
                f"{sd.name}: expected 68 DK cortical ROIs after dropping unknown regions, got {len(df)}. "
                "Inspect annotation naming before proceeding."
            )
        X = df[FEATURES].to_numpy(dtype=np.float32)
        roi_names = df["roi"].to_numpy(dtype=str)

        np.savez_compressed(
            out_dir / f"{sd.name}.npz",
            X=X,
            roi_names=roi_names,
            feature_names=np.array(FEATURES, dtype=str),
        )

        df.insert(0, "fs_id", sd.name)
        all_long.append(df)

    if all_long:
        pd.concat(all_long, ignore_index=True).to_csv(out_dir / "roi_features_long.csv", index=False)
    print(f"Wrote ROI features to {out_dir}")

if __name__ == "__main__":
    main()
