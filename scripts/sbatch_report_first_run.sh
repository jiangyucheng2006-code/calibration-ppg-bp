#!/bin/bash
#SBATCH --job-name=ppg_first_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_first_run_extended_20260815_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"
mkdir -p "$output"

python -m pulsedb_fewshot.report_first_run \
  --controls "$work_root/outputs/event120-v1_calibration_controls_job777/validation_predictions.parquet" \
  --m0 "$work_root/outputs/event120-v1_m0_seed20260813_job774/best_validation_predictions.parquet" \
  --m1 "$work_root/outputs/event120-v1_m1_seed20260813_job775/best_validation_predictions.parquet" \
  --m2 "$work_root/outputs/event120-v1_m2_seed20260813_job776/best_validation_predictions.parquet" \
  --siamese "$work_root/outputs/event120-v1_siamese_seed20260813_job778/best_validation_predictions.parquet" \
  --output-csv "$output/phase5_first_run_extended_metrics.csv" \
  --output-json "$output/phase5_first_run_extended_report.json" \
  --output-markdown "$output/RESULTS_PHASE5_FIRST_RUN_EXTENDED.md"

python - "$output" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

root = Path(sys.argv[1])
metrics = pd.read_csv(root / "phase5_first_run_extended_metrics.csv")
report = json.loads((root / "phase5_first_run_extended_report.json").read_text())
assert len(metrics) == 90
assert set(metrics["N participants"]) == {697}
assert set(metrics["N query events"]) == {103564}
assert report["locked_test_accessed"] is False
assert report["formal_standard_compliance"] == "not established"
print("FIRST_RUN_REPORT_GATE=pass")
PY

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "FIRST_RUN_REPORT_ARCHIVE_IDENTICAL=yes"
echo "FIRST_RUN_REPORT_OUTPUT=$output"
