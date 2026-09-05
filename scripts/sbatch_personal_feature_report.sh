#!/bin/bash
#SBATCH --job-name=ppg_feature_report
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
mode="${1:?mode required}"
seed="${2:?seed required}"
shift 2
ppg_work="/home/$USER/work/ppg_bp"
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ppg_work/envs/train"
export PYTHONPATH="$ppg_project/src"
export PYTHONDONTWRITEBYTECODE=1
ppg_name="personal-feature-mechanisms-v1_${mode}_report_seed${seed}_job${SLURM_JOB_ID}"
ppg_output="$ppg_work/outputs/$ppg_name"
if [[ "$mode" == final ]]; then
  python -m pulsedb_fewshot.personal_feature_report --random-report "$1" --chronological-report "$2" --output "$ppg_output"
else
  arguments=()
  for path in "$@"; do arguments+=(--run "$path"); done
  python -m pulsedb_fewshot.calbased_report "${arguments[@]}" --output "$ppg_output"
fi
mkdir -p "/home/$USER/nas/ppg_bp/outputs/$ppg_name"
rsync -a "$ppg_output/" "/home/$USER/nas/ppg_bp/outputs/$ppg_name/"
cmp "$ppg_output/selection.json" "/home/$USER/nas/ppg_bp/outputs/$ppg_name/selection.json"
