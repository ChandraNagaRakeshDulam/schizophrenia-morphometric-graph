#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np

FEATURES = ["CT", "MC", "Vol", "SD", "SA"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mind-repo", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parcellation", default="aparc")
    a = p.parse_args()

    sys.path.insert(0, str(Path(a.mind_repo).resolve()))
    from MIND import compute_MIND

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sd in sorted(Path(a.subjects_dir).glob(f"{a.dataset}_sub-*")):
        print("MIND:", sd.name)
        M = compute_MIND(str(sd), FEATURES, a.parcellation)
        # Official implementation returns a pandas DataFrame.
        names = np.asarray(M.index.astype(str))
        A = M.to_numpy(dtype=float)
        if A.shape[0] != A.shape[1]:
            raise RuntimeError(f"{sd.name}: non-square MIND matrix {A.shape}")
        np.savez_compressed(
            out_dir / f"{sd.name}.npz",
            A=A.astype(np.float32),
            roi_names=names,
        )
        M.to_csv(out_dir / f"{sd.name}.csv")

    print(f"Wrote MIND matrices to {out_dir}")
    print("Verify that every subject has 68 DK ROIs and consistent ROI names/order before modeling.")

if __name__ == "__main__":
    main()
