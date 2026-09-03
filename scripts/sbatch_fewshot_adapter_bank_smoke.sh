#!/usr/bin/env bash
#SBATCH --job-name=bank-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED OUTPUT_DIR" >&2
  exit 2
fi

seed="$1"
output="$2"
test "$seed" -eq 20260904
work_root="/home/$USER/work/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must identify an immutable source snapshot}"
project_root="$(readlink -f "$project_root")"

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.fewshot_adapter_bank smoke \
  --output "$output" \
  --seed "$seed" \
  --require-cuda
