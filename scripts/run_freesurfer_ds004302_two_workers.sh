#!/usr/bin/env bash
set -u

PROJECT="$HOME/research/schizophrenia-mri/sz_morphometric_graph_starter"

BIDS="$PROJECT/data/raw/ds004302"
SUBJECTS="$PROJECT/data/derivatives/freesurfer_ds004302"
COHORT="$PROJECT/data/metadata/ds004302_external_cohort.csv"
LOGDIR="$PROJECT/logs/ds004302_freesurfer"

LICENSE="/home/rakesh/licenses/freesurfer_license.txt"
IMAGE="freesurfer/freesurfer:7.4.1"

OPENMP=6
WORKERS=2

mkdir -p "$SUBJECTS"
mkdir -p "$LOGDIR"

run_subject () {
    sid="$1"

    fsid="ds004302_${sid}"
    t1="$BIDS/$sid/anat/${sid}_T1w.nii.gz"
    sdir="$SUBJECTS/$fsid"
    logfile="$LOGDIR/${sid}.log"

    echo "[$(date)] CHECK $sid"

    # Skip subjects already completed successfully.
    if [[ -f "$sdir/scripts/recon-all.log" ]] &&
       grep -q "finished without error" "$sdir/scripts/recon-all.log"; then
        echo "[$(date)] SKIP $sid - already complete"
        return 0
    fi

    if [[ ! -f "$t1" ]]; then
        echo "[$(date)] FAIL $sid - missing T1: $t1"
        return 1
    fi

    echo "[$(date)] START $sid"

    # If FreeSurfer was previously initialized, resume without -i.
    if [[ -f "$sdir/mri/orig/001.mgz" ]]; then

        docker run --rm \
          -v "$BIDS:/bids:ro" \
          -v "$SUBJECTS:/subjects" \
          -v "$LICENSE:/usr/local/freesurfer/license.txt:ro" \
          "$IMAGE" \
          recon-all \
          -s "$fsid" \
          -sd /subjects \
          -all \
          -parallel \
          -openmp "$OPENMP" \
          > "$logfile" 2>&1

    else

        docker run --rm \
          -v "$BIDS:/bids:ro" \
          -v "$SUBJECTS:/subjects" \
          -v "$LICENSE:/usr/local/freesurfer/license.txt:ro" \
          "$IMAGE" \
          recon-all \
          -i "/bids/$sid/anat/${sid}_T1w.nii.gz" \
          -s "$fsid" \
          -sd /subjects \
          -all \
          -parallel \
          -openmp "$OPENMP" \
          > "$logfile" 2>&1
    fi

    rc=$?

    if [[ $rc -eq 0 ]] &&
       [[ -f "$sdir/scripts/recon-all.log" ]] &&
       grep -q "finished without error" "$sdir/scripts/recon-all.log"; then
        echo "[$(date)] DONE $sid"
    else
        echo "[$(date)] FAIL $sid return_code=$rc"
    fi
}

export -f run_subject
export PROJECT BIDS SUBJECTS COHORT LOGDIR LICENSE IMAGE OPENMP

echo "=============================================="
echo "ds004302 FreeSurfer batch"
echo "Started: $(date)"
echo "Workers: $WORKERS"
echo "OpenMP per worker: $OPENMP"
echo "=============================================="

python - "$COHORT" <<'PY' |
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

mask = (
    df["include_pre_qc"]
    .astype(str)
    .str.lower()
    .eq("true")
)

for sid in df.loc[mask, "participant_id"]:
    print(sid)
PY
xargs -I{} -P "$WORKERS" bash -c 'run_subject "$1"' _ {}

echo
echo "=============================================="
echo "Batch finished: $(date)"
echo "=============================================="

completed=0
remaining=0

while read -r sid; do
    log="$SUBJECTS/ds004302_${sid}/scripts/recon-all.log"

    if [[ -f "$log" ]] &&
       grep -q "finished without error" "$log"; then
        ((completed+=1))
    else
        ((remaining+=1))
    fi

done < <(
python - "$COHORT" <<'PY'
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])

mask = (
    df["include_pre_qc"]
    .astype(str)
    .str.lower()
    .eq("true")
)

for sid in df.loc[mask, "participant_id"]:
    print(sid)
PY
)

echo "Completed successfully: $completed"
echo "Remaining/failed:       $remaining"
