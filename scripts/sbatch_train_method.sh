#!/bin/bash
#SBATCH --job-name=ppg_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

method="${1:?method is required}"
seed="${2:-20260813}"
population_checkpoint="${3:-}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_${method}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

if [ "$method" = "population" ]; then
  batch_size=256
  episodes_per_epoch=200000
else
  batch_size=64
  episodes_per_epoch=100000
fi

command=(
  python -m pulsedb_fewshot.train
  --method "$method"
  --store-root "$work_root/data/processed/event120-v1"
  --output "$output"
  --seed "$seed"
  --epochs 25
  --patience 5
  --batch-size "$batch_size"
  --workers 4
  --learning-rate 0.0003
  --weight-decay 0.0001
  --episodes-per-epoch "$episodes_per_epoch"
  --require-cuda
)
if [ -n "$population_checkpoint" ]; then
  command+=(--population-checkpoint "$population_checkpoint")
fi
"${command[@]}"

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
echo "TRAINING_COMPLETE=$run_id"
