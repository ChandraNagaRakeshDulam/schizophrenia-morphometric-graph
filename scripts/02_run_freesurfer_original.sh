#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 BIDS_DIR SUBJECTS_DIR FS_LICENSE DATASET_NAME"
  exit 1
fi

BIDS_DIR="$(realpath "$1")"
SUBJECTS_DIR="$(mkdir -p "$2"; realpath "$2")"
FS_LICENSE="$(realpath "$3")"
DATASET="$4"
IMAGE="${FREESURFER_IMAGE:-freesurfer/freesurfer:7.4.1}"

echo "Using FreeSurfer image: ${IMAGE}"
echo "BIDS: ${BIDS_DIR}"
echo "SUBJECTS_DIR: ${SUBJECTS_DIR}"

mapfile -t subjects < <(find "$BIDS_DIR" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -printf '%f\n' | sort)

for sid in "${subjects[@]}"; do
  mapfile -t t1s < <(find "$BIDS_DIR/$sid/anat" -maxdepth 1 -type f -name '*T1w.nii.gz' 2>/dev/null | sort || true)

  if [ "${#t1s[@]}" -ne 1 ]; then
    echo "SKIP ${sid}: expected exactly 1 T1w, found ${#t1s[@]}"
    continue
  fi

  fsid="${DATASET}_${sid}"
  if [ -f "$SUBJECTS_DIR/$fsid/scripts/recon-all.done" ]; then
    echo "DONE already: $fsid"
    continue
  fi

  rel_t1="${t1s[0]#$BIDS_DIR/}"
  echo "RUN $fsid <- $rel_t1"

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
      -all
done
