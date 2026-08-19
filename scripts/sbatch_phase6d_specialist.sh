#!/bin/bash
#SBATCH --job-name=ppg_p6d_expert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

candidate="${1:?candidate is required}"
training_scope="${2:?weighted or hard_only is required}"
hard_weight="${3:?hard participant weight is required}"
seed="${4:-20260813}"
population_checkpoint="${5:?population checkpoint is required}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
labels="$work_root/outputs/event120-v1_phase6d_qgh_routing/oof/participant_risk_labels.parquet"
run_id="event120-v1_phase6d_${candidate}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"
test -f "$labels"
test -f "$population_checkpoint"

extra_args=()
if [[ "$training_scope" == "hard_only" ]]; then
  extra_args+=(--hard-participant-only)
elif [[ "$training_scope" != "weighted" ]]; then
  echo "ERROR: training scope must be weighted or hard_only" >&2
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
  --workers 3 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --episodes-per-epoch 100000 \
  --train-support-policy fixed_first \
  --loss huber \
  --huber-delta 0.5 \
  --episode-sampling participant_balanced \
  --anchor-mode mean \
  --use-quality-gate \
  --ks 5 \
  --participant-risk-labels "$labels" \
  --hard-participant-weight "$hard_weight" \
  "${extra_args[@]}" \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "PHASE6D_SPECIALIST_COMPLETE=$run_id"
