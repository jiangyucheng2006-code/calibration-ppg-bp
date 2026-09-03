#!/bin/bash
#SBATCH --job-name=ppg_ss_combo_final
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

seed="${1:?seed is required}"
random_report="${2:?random report is required}"
chronological_report="${3:?chronological report is required}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
run_id="same-subject-combination-v1_final_report_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
test -d "$project_root"
test ! -w "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.same_subject_combination_report \
  --random-report "$random_report" \
  --chronological-report "$chronological_report" \
  --output "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
cmp -s "$output/selection.json" "$archive_root/outputs/$run_id/selection.json"
echo "SAME_SUBJECT_COMBINATION_FINAL_REPORT_COMPLETE=$run_id"
