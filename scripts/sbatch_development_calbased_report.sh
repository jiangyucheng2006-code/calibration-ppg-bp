#!/bin/bash
#SBATCH --job-name=ppg_cb_report
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

split_mode="${1:?split_mode is required}"
seed="${2:?seed is required}"
shift 2
run_dirs=("$@")
if (( ${#run_dirs[@]} == 0 )); then
  echo "ERROR: at least one run directory is required" >&2
  exit 2
fi

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
run_id="development-calbased-analogue-v1_${split_mode}_screen_report_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
test -d "$project_root"
test ! -w "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

arguments=()
for run_dir in "${run_dirs[@]}"; do
  arguments+=(--run "$run_dir")
done
python -m pulsedb_fewshot.calbased_report \
  "${arguments[@]}" \
  --output "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "DEVELOPMENT_CALBASED_REPORT_COMPLETE=$run_id"
