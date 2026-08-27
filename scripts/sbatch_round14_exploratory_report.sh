#!/bin/bash
#SBATCH --job-name=r14-report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 SEED ANCHOR_RUN SETTING=RUN_DIR [SETTING=RUN_DIR ...]" >&2
  exit 2
fi

seed="$1"
test "$seed" -eq 20260828
anchor_run="$2"
shift 2
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable Round-14 snapshot}"
project_root="$(readlink -f "$project_root")"
output="$work_root/outputs/event120-v1_round14_exploratory_report_seed${seed}_job${SLURM_JOB_ID}"
run_arguments=()
for item in "$@"; do
  run_arguments+=(--run "$item")
done

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.round14_exploratory report-internal \
  "${run_arguments[@]}" \
  --anchor-run "$anchor_run" \
  --output "$output" \
  --expected-seed "$seed"

mkdir -p "$archive_root/outputs"
rsync -a --no-perms --no-owner --no-group "$output/" \
  "$archive_root/outputs/$(basename "$output")/"
diff -qr "$output" "$archive_root/outputs/$(basename "$output")"
