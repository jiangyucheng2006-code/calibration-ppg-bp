#!/bin/bash
#SBATCH --job-name=ppg_r11_pop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

backbone="${1:?backbone is required}"
seed="${2:-20260825}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
folds="$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet"
run_id="event120-v1_round11a_${backbone}_population_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.train \
  --method population \
  --backbone "$backbone" \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$output" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size 256 \
  --workers 4 \
  --episodes-per-epoch 100000 \
  --feature-dim 256 \
  --crossfit-folds "$folds" \
  --crossfit-fit-folds 0 1 2 \
  --crossfit-validation-fold 3 \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
echo "ROUND11A_POPULATION_COMPLETE=$run_id"
