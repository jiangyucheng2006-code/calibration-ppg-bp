#!/bin/bash
#SBATCH --job-name=ppg_r14_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

if [ "$#" -ne 10 ]; then
  echo "ERROR: expected five reference and five candidate SEED=RUN_DIR specifications" >&2
  exit 2
fi
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name the immutable Round-14 snapshot}"
run_id="event120-v1_round14_confirmation_report_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

reference_specs=("${@:1:5}")
candidate_specs=("${@:6:5}")
arguments=()
for specification in "${reference_specs[@]}"; do
  arguments+=(--reference-run "$specification")
done
for specification in "${candidate_specs[@]}"; do
  arguments+=(--candidate-run "$specification")
done

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.round14_confirmation \
  "${arguments[@]}" \
  --expected-seeds 20260827,20260828,20260829,20260830,20260831 \
  --discovery-seed 20260827 \
  --output "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id" >/dev/null
echo "ROUND14_CONFIRMATION_REPORT_COMPLETE=$run_id"
