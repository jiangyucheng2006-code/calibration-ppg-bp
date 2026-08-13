#!/usr/bin/env bash
set -euo pipefail

project="/home/${USER}/work/ppg_bp"
archive="/home/${USER}/nas/ppg_bp"
status="$project/data/manifests/full_acquisition/pulsedb_mimic_005_segmented_status.json"
chunks="$archive/data/raw/PulseDB_v2/downloads/.PulseDB_MIMIC.zip.005.chunks"
target="$archive/data/raw/PulseDB_v2/downloads/box_parts/PulseDB_MIMIC.zip.005"
job="$(squeue -u "$USER" -h -n ppg_finish_005 -o '%i|%T|%M|%R' | head -n 1 || true)"

echo "TIME=$(date -Is)"
echo "JOB=${job:-not_in_queue}"
stat -c 'TARGET_BYTES=%s' "$target"
if [[ -f "$status" ]]; then
  "$project/envs/train/bin/python" - "$status" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "phase", "completed_chunk_bytes", "expected_bytes", "percent",
    "workers", "chunk_size", "chunk_count", "state_counts", "errors",
    "started_at_utc", "updated_at_utc",
):
    print(f"{key.upper()}={d.get(key)}")
PY
else
  echo "STATUS=not_created"
fi
if [[ -d "$chunks" ]]; then
  find "$chunks" -maxdepth 1 -type f -printf '%s\n' \
    | awk '{sum += $1; count += 1} END {printf "CHUNK_FILES=%d\nCHUNK_DISK_BYTES=%.0f\n", count, sum}'
else
  echo "CHUNK_DIRECTORY=absent"
fi
job_id="${job%%|*}"
if [[ "$job_id" =~ ^[0-9]+$ ]]; then
  stdout="$project/logs/pulsedb_finish_005_${job_id}.out"
  stderr="$project/logs/pulsedb_finish_005_${job_id}.err"
  [[ -f "$stdout" ]] && tail -n 12 "$stdout"
  [[ -f "$stderr" ]] && stat -c 'STDERR_BYTES=%s' "$stderr"
fi
