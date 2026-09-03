#!/bin/bash
#SBATCH --job-name=r14-prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 POPULATION_RUN QGH_RUN SEED" >&2
  exit 2
fi

population_run="$1"
qgh_run="$2"
seed="$3"
test "$seed" -eq 20260828
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable Round-14 snapshot}"
project_root="$(readlink -f "$project_root")"
output="$work_root/outputs/event120-v1_round14_exploratory_cache_seed${seed}_job${SLURM_JOB_ID}"

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.round14_exploratory prepare-cache \
  --store-root "$work_root/data/processed/event120-v1" \
  --folds "$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet" \
  --population-run "$population_run" \
  --qgh-run "$qgh_run" \
  --output "$output" \
  --seed "$seed" \
  --batch-size 256 \
  --require-cuda

mkdir -p "$archive_root/outputs"
rsync -a --no-perms --no-owner --no-group "$output/" \
  "$archive_root/outputs/$(basename "$output")/"
diff -qr "$output" "$archive_root/outputs/$(basename "$output")"
