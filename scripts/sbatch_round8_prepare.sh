#!/bin/bash
#SBATCH --job-name=ppg_r8_prepare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
output="$work_root/outputs/event120-v1_round8_prepared_job${SLURM_JOB_ID}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.round8_calibration_relative prepare \
  --store-root "$work_root/data/processed/event120-v1" \
  --population-checkpoint "$work_root/outputs/event120-v1_repeat5-v1_population_seed20260813_job782/best.pt" \
  --train-features "$work_root/outputs/event120-v1_phase6d_qgh_routing/oof/oof_risk_features.parquet" \
  --validation-features "$work_root/outputs/event120-v1_phase6e_continuous/validation_features.parquet" \
  --query-embeddings "$work_root/outputs/event120-v1_phase6e_continuous/waveform_embeddings.parquet" \
  --output "$output"

mkdir -p "$archive_root/outputs/$(basename "$output")"
rsync -a "$output/" "$archive_root/outputs/$(basename "$output")/"
echo "ROUND8_PREPARATION_COMPLETE=$output"
