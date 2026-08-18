#!/bin/bash
#SBATCH --job-name=ppg_tail_folds
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

seed="${1:-20260818}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
pipeline_root="$work_root/outputs/event120-v1_tailrisk-v1"
output="$pipeline_root/folds"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.tail_risk prepare-folds \
  --store-root "$work_root/data/processed/event120-v1" \
  --output-dir "$output" \
  --n-folds 5 \
  --seed "$seed"

mkdir -p "$archive_root/outputs/event120-v1_tailrisk-v1/folds"
rsync -a "$output/" "$archive_root/outputs/event120-v1_tailrisk-v1/folds/"
diff -qr "$output" "$archive_root/outputs/event120-v1_tailrisk-v1/folds"
echo "TAILRISK_FOLDS_COMPLETE=yes"
