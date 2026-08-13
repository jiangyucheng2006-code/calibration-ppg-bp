#!/bin/bash
#SBATCH --job-name=ppg_full_cohort
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
nas_root="/home/$USER/nas/ppg_bp"
project="$work_root/code/pulsedb_fewshot"
python="$work_root/envs/train/bin/python"
data_root="$nas_root/data/raw/PulseDB_v2/extraction_staging_20260811/Segment_Files"
output_root="$work_root/data/manifests/pulsedb_v2_full_cohort"
archive_root="$nas_root/data/manifests/pulsedb_v2_full_cohort"
event_root="$work_root/data/events/pulsedb_v2_development_feasibility"
event_archive="$nas_root/data/events/pulsedb_v2_development_feasibility"

echo "===== PULSEDB V2 FULL COHORT AUDIT ====="
echo "TIME=$(date -Is)"
echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_ROOT=$data_root"
echo "OUTPUT_ROOT=$output_root"

cd "$project"
"$python" -m pytest -q

"$python" -m pulsedb_fewshot.full_cohort \
  --data-root "$data_root" \
  --output "$output_root" \
  --workers 4 \
  --seed 20260809

"$python" -m pulsedb_fewshot.development_events \
  --development-segments "$output_root/development_segments.parquet" \
  --subject-splits "$output_root/subject_splits.csv" \
  --output "$event_root" \
  --widths 60 120 300 \
  --min-query-events 5

mkdir -p "$archive_root" "$event_archive"
rsync -a --delete "$output_root/" "$archive_root/"
rsync -a --delete "$event_root/" "$event_archive/"

"$python" - "$output_root" "$archive_root" "$event_root" "$event_archive" <<'PY'
from pathlib import Path
import hashlib
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


work, archive, event_work, event_archive = map(Path, sys.argv[1:])
for left_root, right_root in ((work, archive), (event_work, event_archive)):
    left = sorted(path.relative_to(left_root) for path in left_root.rglob("*") if path.is_file())
    right = sorted(path.relative_to(right_root) for path in right_root.rglob("*") if path.is_file())
    if left != right:
        raise SystemExit(f"archive inventory mismatch: {left_root} vs {right_root}")
    for relative in left:
        if digest(left_root / relative) != digest(right_root / relative):
            raise SystemExit(f"archive content mismatch: {relative}")
print("ARCHIVE_TREES_IDENTICAL=yes")
PY

echo "PULSEDB_V2_FULL_COHORT_COMPLETE=yes"
echo "DEVELOPMENT_EVENT_FEASIBILITY_COMPLETE=yes"
echo "EVENT_SPACING_SELECTED=no"
echo "LOCKED_TEST_EVENTIZED=no"
