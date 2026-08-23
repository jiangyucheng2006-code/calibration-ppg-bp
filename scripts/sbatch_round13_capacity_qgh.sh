#!/bin/bash
#SBATCH --job-name=ppg_r13_qgh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

backbone="${1:?backbone is required}"
population_run="${2:?population run directory is required}"
seed="${3:-20260827}"
batch_size="${4:-64}"
accumulation_steps="${5:-1}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:-$work_root/code/pulsedb_fewshot}"
folds="$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet"
run_id="event120-v1_round13_${backbone}_qgh_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.train \
  --method m0 \
  --backbone "$backbone" \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$output" \
  --population-checkpoint "$population_run/best.pt" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size "$batch_size" \
  --gradient-accumulation-steps "$accumulation_steps" \
  --workers 4 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --train-support-policy fixed_first \
  --loss huber \
  --huber-delta 0.5 \
  --use-quality-gate \
  --ks 5 \
  --episodes-per-epoch 99968 \
  --crossfit-folds "$folds" \
  --crossfit-fit-folds 0 1 2 \
  --crossfit-validation-fold 3 \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
echo "ROUND13_QGH_COMPLETE=$run_id"
