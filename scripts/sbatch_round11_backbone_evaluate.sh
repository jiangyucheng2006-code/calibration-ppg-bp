#!/bin/bash
#SBATCH --job-name=ppg_r11_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

backbone="${1:?backbone is required}"
population_run="${2:?population run directory is required}"
qgh_run="${3:?QGH run directory is required}"
seed="${4:-20260825}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
folds="$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet"
run_id="event120-v1_round11a_${backbone}_evaluation_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.round11_backbones evaluate \
  --backbone "$backbone" \
  --store-root "$work_root/data/processed/event120-v1" \
  --folds "$folds" \
  --population-run "$population_run" \
  --qgh-run "$qgh_run" \
  --output "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "ROUND11A_EVALUATION_COMPLETE=$run_id"
