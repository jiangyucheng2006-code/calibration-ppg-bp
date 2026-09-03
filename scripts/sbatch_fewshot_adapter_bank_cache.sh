#!/usr/bin/env bash
#SBATCH --job-name=bank-cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 POPULATION_RUN SEED" >&2
  exit 2
fi

population_run="$1"
seed="$2"
test "$seed" -eq 20260904
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must identify an immutable source snapshot}"
project_root="$(readlink -f "$project_root")"
output="$work_root/outputs/event120-v1_fewshot_adapter_bank_cache_seed${seed}_job${SLURM_JOB_ID}"

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.fewshot_adapter_bank prepare-cache \
  --store-root "$work_root/data/processed/event120-v1" \
  --folds "$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet" \
  --population-run "$population_run" \
  --output "$output" \
  --batch-size 1024 \
  --require-cuda

mkdir -p "$archive_root/outputs"
rsync -a --no-perms --no-owner --no-group "$output/" \
  "$archive_root/outputs/$(basename "$output")/"
diff -qr "$output" "$archive_root/outputs/$(basename "$output")"
