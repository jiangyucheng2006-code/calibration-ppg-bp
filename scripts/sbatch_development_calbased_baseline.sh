#!/bin/bash
#SBATCH --job-name=ppg_cb_mean
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

split_mode="${1:?split_mode is required}"
seed="${2:-20260828}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
store_root="$work_root/data/processed/development-calbased-analogue-v1"
run_id="development-calbased-analogue-v1_${split_mode}_subject_train_mean_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
test -d "$project_root"
test ! -w "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.calbased_train \
  --candidate subject_train_mean \
  --store-root "$store_root" \
  --split-mode "$split_mode" \
  --output "$output" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "DEVELOPMENT_CALBASED_BASELINE_COMPLETE=$run_id"
