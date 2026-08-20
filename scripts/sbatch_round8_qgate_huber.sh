#!/bin/bash
#SBATCH --job-name=ppg_r8_qgh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

candidate="${1:?candidate name is required}"
population_checkpoint="${2:?population checkpoint is required}"
feature_dim="${3:?feature dimension is required}"
demographic_mode="${4:-none}"
seed="${5:-20260822}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_round8_${candidate}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

extra_args=()
if [[ "$demographic_mode" != "none" ]]; then
  demographics="$work_root/data/manifests/phase6_demographics/participant_demographics_clean.parquet"
  test -f "$demographics"
  extra_args+=(
    --use-demographics
    --demographics-path "$demographics"
    --demographic-mode "$demographic_mode"
  )
fi

python -m pulsedb_fewshot.train \
  --method m0 \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$output" \
  --population-checkpoint "$population_checkpoint" \
  --feature-dim "$feature_dim" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size 64 \
  --workers 4 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --episodes-per-epoch 100000 \
  --train-support-policy fixed_first \
  --loss huber \
  --huber-delta 0.5 \
  --episode-sampling participant_balanced \
  --anchor-mode mean \
  --use-quality-gate \
  "${extra_args[@]}" \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
echo "ROUND8_QGATE_HUBER_COMPLETE=$run_id"
