#!/bin/bash
#SBATCH --job-name=ppg_calibration
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

population_job="${1:?population job ID is required}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
population_checkpoint="$work_root/outputs/event120-v1_population_seed20260813_job${population_job}/best.pt"
run_id="event120-v1_calibration_controls_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.evaluate_calibration \
  --store-root "$work_root/data/processed/event120-v1" \
  --population-checkpoint "$population_checkpoint" \
  --output "$output" \
  --seed 20260813 \
  --steps 20 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --lora-rank 4 \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "CALIBRATION_CONTROLS_COMPLETE=$run_id"
