#!/bin/bash
#SBATCH --job-name=ppg_r10_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

method="${1:?Round-10 method is required}"
prepared="${2:?prepared directory is required}"
population_run="${3:?population run directory is required}"
seed="${4:-20260824}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_round10_internal_${method}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"
batch_size=4
if [ "$method" = "last_two_blocks" ]; then
  batch_size=2
elif [ "$method" = "full_encoder" ]; then
  batch_size=1
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.round10_end_to_end train \
  --prepared "$prepared" \
  --population-checkpoint "$population_run/best.pt" \
  --output "$output" \
  --method "$method" \
  --seed "$seed" \
  --batch-size "$batch_size"

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
echo "ROUND10_INTERNAL_CANDIDATE_COMPLETE=$run_id"
