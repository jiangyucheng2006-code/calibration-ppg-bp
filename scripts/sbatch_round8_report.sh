#!/bin/bash
#SBATCH --job-name=ppg_r8_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

seed="${1:?seed is required}"
shift
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_round8_report_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

command=(
  python -m pulsedb_fewshot.report_round8
  --reference-name "Quality Gate + Huber (256-D)"
  --reference-dir "$work_root/outputs/event120-v1_phase6b_qgate_huber_seed20260813_job841"
  --similarity-path "$work_root/outputs/event120-v1_meta_validation_10s_beat_similarity_job946/window_similarity_private.parquet"
  --output "$output"
  --expected-seed "$seed"
)
for specification in "$@"; do
  command+=(--candidate "$specification")
done
"${command[@]}"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "ROUND8_REPORT_COMPLETE=$run_id"
