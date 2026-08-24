#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage:"
  echo "$0 COHORT_CSV BIDS_DIR SUBJECTS_DIR FS_LICENSE DATASET_NAME"
  exit 1
fi

COHORT_CSV="$(realpath "$1")"
BIDS_DIR="$(realpath "$2")"
SUBJECTS_DIR="$(mkdir -p "$3"; realpath "$3")"
FS_LICENSE="$(realpath "$4")"
DATASET="$5"

IMAGE="${FREESURFER_IMAGE:-freesurfer/freesurfer:7.4.1}"
OPENMP="${OPENMP:-8}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"

LOG_DIR="${FS_BATCH_LOG_DIR:-$(pwd)/logs/freesurfer_batch}"
mkdir -p "$LOG_DIR"

STATUS_FILE="$LOG_DIR/status.csv"

if [ ! -f "$STATUS_FILE" ]; then
  echo "subject_id,fs_id,status,exit_code,timestamp" > "$STATUS_FILE"
fi

echo "========================================"
echo "FreeSurfer cohort batch"
echo "========================================"
echo "Image:        $IMAGE"
echo "Dataset:      $DATASET"
echo "BIDS:         $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "OpenMP:       $OPENMP"
echo "Max subjects: $MAX_SUBJECTS"
echo "========================================"

mapfile -t subjects < <(
python3 - "$COHORT_CSV" <<'PY'
import csv
import sys

path = sys.argv[1]

with open(path, newline="") as f:
    rows = csv.DictReader(f)

    for r in rows:
        diagnosis = r.get("diagnosis", "").strip()
        n_t1 = r.get("n_t1", "").strip()
        include = r.get("include_pre_qc", "").strip().lower()

        if (
            diagnosis in {"CONTROL", "SCHZ"}
            and n_t1 == "1"
            and include in {"true", "1", "yes"}
        ):
            print(r["subject_id"].strip())
PY
)

echo "Eligible subjects: ${#subjects[@]}"

started=0
success=0
failed=0
skipped=0

for sid in "${subjects[@]}"; do

  fsid="${DATASET}_${sid}"
  subject_dir="$SUBJECTS_DIR/$fsid"
  log="$LOG_DIR/${fsid}.log"

  # --------------------------------------------------
  # Already successfully completed?
  # --------------------------------------------------
  if [ -f "$subject_dir/scripts/recon-all.done" ] || \
     grep -q "finished without error" \
       "$subject_dir/scripts/recon-all.log" 2>/dev/null; then

    echo "SKIP completed: $fsid"
    skipped=$((skipped + 1))
    continue
  fi

  # Limit number of NEW runs when testing.
  if [ "$MAX_SUBJECTS" -gt 0 ] && [ "$started" -ge "$MAX_SUBJECTS" ]; then
    echo "Reached MAX_SUBJECTS=$MAX_SUBJECTS"
    break
  fi

  # --------------------------------------------------
  # Locate exactly one T1.
  # Glob works for both normal files and git-annex symlinks.
  # --------------------------------------------------
  shopt -s nullglob
  t1s=("$BIDS_DIR/$sid/anat/"*T1w.nii.gz)
  shopt -u nullglob

  if [ "${#t1s[@]}" -ne 1 ]; then
    echo "FAIL $sid: expected exactly 1 T1w, found ${#t1s[@]}"
    echo "$sid,$fsid,T1_COUNT_ERROR,1,$(date -Iseconds)" >> "$STATUS_FILE"
    failed=$((failed + 1))
    continue
  fi

  t1="${t1s[0]}"
  rel_t1="${t1#$BIDS_DIR/}"

  # --------------------------------------------------
  # Retrieve T1 from git-annex if not stored locally.
  # --------------------------------------------------
  if [ ! -e "$t1" ]; then
    echo "GET T1: $sid"

    if ! git -C "$BIDS_DIR" annex get "$rel_t1"; then
      echo "FAIL $sid: git-annex retrieval failed"
      echo "$sid,$fsid,ANNEX_FAILED,1,$(date -Iseconds)" >> "$STATUS_FILE"
      failed=$((failed + 1))
      continue
    fi
  fi

  if [ ! -e "$t1" ]; then
    echo "FAIL $sid: T1 is still unavailable"
    echo "$sid,$fsid,T1_UNAVAILABLE,1,$(date -Iseconds)" >> "$STATUS_FILE"
    failed=$((failed + 1))
    continue
  fi

  echo
  echo "========================================"
  echo "RUN: $fsid"
  echo "T1:  $rel_t1"
  echo "========================================"

  started=$((started + 1))

  # --------------------------------------------------
  # Resume an incomplete subject OR start a new one.
  # --------------------------------------------------
  if [ -d "$subject_dir" ]; then

    echo "Existing incomplete subject detected."
    echo "Removing stale FreeSurfer IsRunning locks."

    docker run --rm \
      -v "$SUBJECTS_DIR":/subjects \
      "$IMAGE" \
      bash -lc "rm -f /subjects/$fsid/scripts/IsRunning*"

    docker run --rm \
      -e FS_LICENSE=/license.txt \
      -v "$FS_LICENSE":/license.txt:ro \
      -v "$BIDS_DIR":/bids:ro \
      -v "$SUBJECTS_DIR":/subjects \
      "$IMAGE" \
      recon-all \
        -s "$fsid" \
        -sd /subjects \
        -all \
        -parallel \
        -openmp "$OPENMP" \
        2>&1 | tee "$log"

    exit_code=${PIPESTATUS[0]}

  else

    docker run --rm \
      -e FS_LICENSE=/license.txt \
      -v "$FS_LICENSE":/license.txt:ro \
      -v "$BIDS_DIR":/bids:ro \
      -v "$SUBJECTS_DIR":/subjects \
      "$IMAGE" \
      recon-all \
        -i "/bids/$rel_t1" \
        -s "$fsid" \
        -sd /subjects \
        -all \
        -parallel \
        -openmp "$OPENMP" \
        2>&1 | tee "$log"

    exit_code=${PIPESTATUS[0]}

  fi

  # --------------------------------------------------
  # Verify real completion.
  # --------------------------------------------------
  if [ "$exit_code" -eq 0 ] && \
     grep -q "finished without error" \
       "$subject_dir/scripts/recon-all.log" 2>/dev/null; then

    echo "SUCCESS: $fsid"
    echo "$sid,$fsid,SUCCESS,0,$(date -Iseconds)" >> "$STATUS_FILE"
    success=$((success + 1))

  else

    echo "FAILED: $fsid"
    echo "$sid,$fsid,FAILED,$exit_code,$(date -Iseconds)" >> "$STATUS_FILE"
    failed=$((failed + 1))

  fi

done

echo
echo "========================================"
echo "BATCH SUMMARY"
echo "========================================"
echo "Eligible: ${#subjects[@]}"
echo "Started:  $started"
echo "Success:  $success"
echo "Failed:   $failed"
echo "Skipped:  $skipped"
echo
echo "Status:"
echo "$STATUS_FILE"
