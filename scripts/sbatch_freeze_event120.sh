#!/bin/bash
#SBATCH --job-name=ppg_event120
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
input_root="$work_root/data/manifests/pulsedb_v2_full_cohort"
output_root="$work_root/data/events/event120-v1"
archive_output="$archive_root/data/events/event120-v1"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pytest
python -m pulsedb_fewshot.frozen_protocol \
  --full-segments "$input_root/pulsedb_v2_full_segment_index.parquet" \
  --subject-splits "$input_root/subject_splits.csv" \
  --output "$output_root"

python - <<'PY'
from pathlib import Path
import json

root = Path("/home") / Path.home().name / "work/ppg_bp/data/events/event120-v1"
protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
audit = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
assert protocol["status"] == "pass"
assert protocol["protocol_id"] == "event120-v1"
assert audit["status"] == "pass"
assert audit["query_labels_in_model_input"] is False
print("EVENT120_PROTOCOL_GATE=pass")
print(f"DEVELOPMENT_ELIGIBLE_SUBJECTS={audit['development_subjects']}")
print(f"LOCKED_ELIGIBLE_SUBJECTS={audit['locked_eligible_subjects']}")
print(f"LOCKED_QUERY_TARGETS={audit['locked_query_targets']}")
PY

mkdir -p "$archive_output"
rsync -a --delete "$output_root/" "$archive_output/"
diff -qr "$output_root" "$archive_output"
echo "EVENT120_ARCHIVE_IDENTICAL=yes"
echo "EVENT120_FREEZE_COMPLETE=yes"
