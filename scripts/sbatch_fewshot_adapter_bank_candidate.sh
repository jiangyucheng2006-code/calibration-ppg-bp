#!/usr/bin/env bash
#SBATCH --job-name=bank-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 SETTING BASIS_COUNT ROUTING_MODE CACHE_DIR POPULATION_RUN SEED" >&2
  exit 2
fi

setting="$1"
basis_count="$2"
routing_mode="$3"
cache_dir="$4"
population_run="$5"
seed="$6"
test "$seed" -eq 20260904
if [[ "$setting" == "m0_reference" ]]; then
  [[ "$basis_count" == "0" && "$routing_mode" == "none" ]] || exit 2
else
  [[ "$setting" == "bank$(printf '%02d' "$basis_count")_${routing_mode}" ]] || exit 2
  [[ "$basis_count" =~ ^(5|10|15|20|25|30)$ ]] || exit 2
  [[ "$routing_mode" =~ ^(top5|dense)$ ]] || exit 2
fi

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must identify an immutable source snapshot}"
project_root="$(readlink -f "$project_root")"
output="$work_root/outputs/event120-v1_${setting}_seed${seed}_job${SLURM_JOB_ID}"

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.fewshot_adapter_bank train \
  --prepared "$cache_dir" \
  --population-run "$population_run" \
  --output "$output" \
  --basis-count "$basis_count" \
  --routing-mode "$routing_mode" \
  --seed "$seed" \
  --batch-size 2048 \
  --evaluation-batch-size 8192 \
  --episodes-per-epoch 99968 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --patience 8 \
  --require-cuda

mkdir -p "$archive_root/outputs"
rsync -a --no-perms --no-owner --no-group "$output/" \
  "$archive_root/outputs/$(basename "$output")/"
diff -qr "$output" "$archive_root/outputs/$(basename "$output")"
