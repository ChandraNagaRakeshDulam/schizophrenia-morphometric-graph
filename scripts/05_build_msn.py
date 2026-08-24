#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roi-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    in_dir = Path(a.roi_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(in_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        X = d["X"].astype(float)  # [68, 5]
        names = d["roi_names"].astype(str)

        # Within-subject feature standardization across cortical regions.
        # This uses only the current subject, so it creates no cross-subject leakage.
        Z = StandardScaler().fit_transform(X)

        # Correlation between regional 5-feature morphometric profiles.
        A = np.corrcoef(Z)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 1.0)

        np.savez_compressed(out_dir / f.name, A=A.astype(np.float32), roi_names=names)

    print(f"Wrote MSN matrices to {out_dir}")

if __name__ == "__main__":
    main()
