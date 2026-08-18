#!/bin/bash
#SBATCH --job-name=ppg_p6b_tail
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

candidate="${1:?candidate is required}"
seed="${2:-20260813}"
population_checkpoint="${3:?population checkpoint is required}"
loss_name="${4:?loss name is required}"
quality_gate="${5:?quality-gate flag is required}"
tail_objective="${6:?tail objective is required}"

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_phase6b_${candidate}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

extra_args=()
if [[ "$quality_gate" == "yes" ]]; then
  extra_args+=(--use-quality-gate)
elif [[ "$quality_gate" != "no" ]]; then
  echo "ERROR: quality-gate flag must be yes or no" >&2
  exit 1
fi

python -m pulsedb_fewshot.train \
  --method m0 \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$output" \
  --population-checkpoint "$population_checkpoint" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size 64 \
  --workers 4 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --episodes-per-epoch 100000 \
  --train-support-policy fixed_first \
  --loss "$loss_name" \
  --huber-delta 0.5 \
  --episode-sampling participant_balanced \
  --anchor-mode mean \
  --tail-objective "$tail_objective" \
  --tail-fraction 0.30 \
  --tail-weight 0.50 \
  "${extra_args[@]}" \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
echo "PHASE6B_TAIL_TRAINING_COMPLETE=$run_id"
