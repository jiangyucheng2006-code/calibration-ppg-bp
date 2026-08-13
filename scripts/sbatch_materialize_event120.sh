#!/bin/bash
#SBATCH --job-name=ppg_materialize
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
events_root="$work_root/data/events/event120-v1"
output_root="$work_root/data/processed/event120-v1"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pytest -k 'not models'
if [ -e "$output_root" ]; then
  echo "ERROR: refusing to overwrite existing materialization root: $output_root" >&2
  exit 1
fi
python -m pulsedb_fewshot.materialize \
  --development "$events_root/development_episodes.parquet" \
  --locked-inputs "$events_root/locked_model_inputs.parquet" \
  --output "$output_root" \
  --development-shards 32 \
  --locked-shards 8 \
  --workers "$SLURM_CPUS_PER_TASK"

test -s "$output_root/materialization.json"
echo "EVENT120_MATERIALIZATION_COMPLETE=yes"
