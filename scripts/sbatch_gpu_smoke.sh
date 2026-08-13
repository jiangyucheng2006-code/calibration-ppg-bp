#!/bin/bash
#SBATCH --job-name=ppg_gpu_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
smoke_root="$work_root/outputs/gpu_smoke/job${SLURM_JOB_ID}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m pytest
test ! -e "$smoke_root"
python -m pulsedb_fewshot.train \
  --method population \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$smoke_root/population" \
  --seed 20260813 \
  --epochs 1 \
  --patience 1 \
  --batch-size 16 \
  --workers 2 \
  --episodes-per-epoch 64 \
  --max-train-examples 64 \
  --max-validation-examples 64 \
  --require-cuda

test -s "$smoke_root/population/best.pt"
mkdir -p "$archive_root/outputs/gpu_smoke/job${SLURM_JOB_ID}"
rsync -a "$smoke_root/" "$archive_root/outputs/gpu_smoke/job${SLURM_JOB_ID}/"
echo "GPU_SMOKE_COMPLETE=yes"
