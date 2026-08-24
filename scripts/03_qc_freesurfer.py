#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = [
    "scripts/recon-all.done",
    "mri/aseg.mgz",
    "surf/lh.white", "surf/rh.white",
    "surf/lh.pial", "surf/rh.pial",
    "surf/lh.thickness", "surf/rh.thickness",
    "surf/lh.curv", "surf/rh.curv",
    "surf/lh.sulc", "surf/rh.sulc",
    "surf/lh.area", "surf/rh.area",
    "surf/lh.volume", "surf/rh.volume",
    "label/lh.aparc.annot", "label/rh.aparc.annot",
    "stats/lh.aparc.stats", "stats/rh.aparc.stats",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    root = Path(a.subjects_dir)
    rows = []
    for sd in sorted(root.glob(f"{a.dataset}_sub-*")):
        missing = [x for x in REQUIRED if not (sd / x).exists()]
        log = sd / "scripts" / "recon-all.log"
        log_text = log.read_text(errors="ignore") if log.exists() else ""
        explicit_error = "recon-all -s" in log_text and "exited with ERRORS" in log_text
        rows.append({
            "fs_id": sd.name,
            "subject_id": sd.name[len(a.dataset) + 1:],
            "required_files_ok": len(missing) == 0,
            "missing_files": ";".join(missing),
            "log_reports_error": explicit_error,
            "manual_surface_qc": "PENDING",
            "final_qc_pass": False,
        })

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["required_files_ok", "log_reports_error"]].value_counts(dropna=False))
    print(f"Wrote {out}")
    print("IMPORTANT: visually inspect white/pial surfaces in Freeview; set manual_surface_qc and final_qc_pass manually.")

if __name__ == "__main__":
    main()
