#!/bin/bash
#SBATCH --job-name=ppg_tail_route
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

candidate="${1:?candidate name is required}"
expert_run="${2:?expert run directory is required}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
reference="$work_root/outputs/event120-v1_phase6_fixed_first_seed20260813_job818"
risk_checkpoint="$work_root/outputs/event120-v1_tailrisk-v1/risk_classifier/risk_model.pt"
run_id="event120-v1_tailrisk_route_${candidate}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"
test -f "$reference/run.json"
test -f "$expert_run/run.json"
test -f "$risk_checkpoint"

python -m pulsedb_fewshot.tail_risk evaluate-routing \
  --store-root "$work_root/data/processed/event120-v1" \
  --reference-run "$reference" \
  --expert-run "$expert_run" \
  --risk-checkpoint "$risk_checkpoint" \
  --output-dir "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "TAILRISK_ROUTING_COMPLETE=$run_id"
