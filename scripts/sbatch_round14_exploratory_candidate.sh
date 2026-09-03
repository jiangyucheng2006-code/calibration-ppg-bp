#!/bin/bash
#SBATCH --job-name=r14-method
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 METHOD CACHE_DIR SEED" >&2
  exit 2
fi

method="$1"
cache_dir="$2"
seed="$3"
test "$seed" -eq 20260828
case "$method" in
  calibration_relative|calibration_relative_standards|calibration_relative_groupdro) ;;
  *) echo "unsupported Round-14 method: $method" >&2; exit 2 ;;
esac

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable Round-14 snapshot}"
project_root="$(readlink -f "$project_root")"
output="$work_root/outputs/event120-v1_round14_${method}_seed${seed}_job${SLURM_JOB_ID}"

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.round14_exploratory train-candidate \
  --prepared "$cache_dir" \
  --output "$output" \
  --method "$method" \
  --seed "$seed" \
  --batch-size 12 \
  --require-cuda

mkdir -p "$archive_root/outputs"
rsync -a --no-perms --no-owner --no-group "$output/" \
  "$archive_root/outputs/$(basename "$output")/"
diff -qr "$output" "$archive_root/outputs/$(basename "$output")"
