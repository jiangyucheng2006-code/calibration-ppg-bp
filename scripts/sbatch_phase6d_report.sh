#!/bin/bash
#SBATCH --job-name=ppg_p6d_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

shift_args=("$@")
[[ "${#shift_args[@]}" -ge 2 ]] || { echo "ERROR: candidate=run arguments are required" >&2; exit 1; }
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
pipeline_root="$work_root/outputs/event120-v1_phase6d_qgh_routing"
general="$work_root/outputs/event120-v1_phase6b_qgate_huber_seed20260813_job841"
risk_checkpoint="$pipeline_root/risk_classifier/risk_model.pt"
run_id="event120-v1_phase6d_complete_pipeline_report_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"
test -f "$general/run.json"
test -f "$risk_checkpoint"

command=(python -m pulsedb_fewshot.report_phase6d_pipeline
  --store-root "$work_root/data/processed/event120-v1"
  --general-run "$general"
  --risk-checkpoint "$risk_checkpoint"
  --output-dir "$output"
  --tail-fraction 0.30)
for expert in "${shift_args[@]}"; do
  command+=(--expert "$expert")
done
"${command[@]}"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "PHASE6D_REPORT_COMPLETE=$run_id"
