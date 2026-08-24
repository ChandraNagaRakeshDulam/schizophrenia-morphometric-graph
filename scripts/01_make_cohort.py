#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bids-dir", required=True)
    p.add_argument("--participants", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--diagnosis-column", required=True)
    p.add_argument("--case-values", nargs="+", required=True)
    p.add_argument("--control-values", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    bids = Path(a.bids_dir)
    df = pd.read_csv(a.participants, sep="\t")

    print("participants.tsv columns:")
    print(list(df.columns))
    if a.diagnosis_column not in df.columns:
        raise SystemExit(
            f"Diagnosis column {a.diagnosis_column!r} not found. "
            "Inspect the printed columns and rerun with the correct column."
        )
    if "participant_id" not in df.columns:
        raise SystemExit("BIDS participants.tsv must contain participant_id.")

    case_values = {x.strip().lower() for x in a.case_values}
    control_values = {x.strip().lower() for x in a.control_values}

    def map_dx(x):
        z = str(x).strip().lower()
        if z in case_values:
            return 1
        if z in control_values:
            return 0
        return pd.NA

    df["diagnosis_binary"] = df[a.diagnosis_column].map(map_dx)
    df["dataset"] = a.dataset
    df["subject_id"] = df["participant_id"].astype(str)
    df["fs_id"] = a.dataset + "_" + df["subject_id"]

    t1_map = {}
    for sid in df["subject_id"]:
        anat = bids / sid / "anat"
        hits = sorted(anat.glob("*T1w.nii.gz")) if anat.exists() else []
        t1_map[sid] = hits

    df["n_t1"] = df["subject_id"].map(lambda x: len(t1_map[x]))
    df["t1_path"] = df["subject_id"].map(
        lambda x: str(t1_map[x][0]) if len(t1_map[x]) == 1 else ""
    )
    df["include_pre_qc"] = (
        df["diagnosis_binary"].notna() & (df["n_t1"] == 1)
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\nDiagnosis values found:")
    print(df[a.diagnosis_column].value_counts(dropna=False))
    print("\nBinary cohort counts:")
    print(df.loc[df["include_pre_qc"], "diagnosis_binary"].value_counts(dropna=False))
    print("\nT1 counts:")
    print(df["n_t1"].value_counts(dropna=False).sort_index())
    print(f"\nWrote {out}")
    print("STOP if any diagnostic mapping or T1 count looks wrong.")

if __name__ == "__main__":
    main()
