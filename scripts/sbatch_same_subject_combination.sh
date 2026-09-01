#!/bin/bash
#SBATCH --job-name=ppg_ss_combo
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

candidate="${1:?candidate is required}"
split_mode="${2:?split_mode is required}"
seed="${3:-20260902}"
batch_size="${4:-64}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
store_root="$work_root/data/processed/development-calbased-analogue-v1"
run_id="same-subject-combination-v1_${split_mode}_${candidate}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
test -d "$project_root"
test ! -w "$project_root"
test -f "$store_root/materialization.json"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.same_subject_combination_train \
  --candidate "$candidate" \
  --store-root "$store_root" \
  --split-mode "$split_mode" \
  --output "$output" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size "$batch_size" \
  --workers 4 \
  --examples-per-epoch 200000 \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
cmp -s "$output/run.json" "$archive_root/outputs/$run_id/run.json"
echo "SAME_SUBJECT_COMBINATION_COMPLETE=$run_id"
