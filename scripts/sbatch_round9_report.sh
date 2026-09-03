#!/bin/bash
#SBATCH --job-name=ppg_r9_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

seed="${1:?seed is required}"
shift
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_round9_internal_report_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

command=(
  python -m pulsedb_fewshot.round9_refinement report-internal
  --reference "R9-0 R8 architecture reference"
  --output "$output"
  --expected-seed "$seed"
)
for specification in "$@"; do
  command+=(--run "$specification")
done
"${command[@]}"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "ROUND9_INTERNAL_REPORT_COMPLETE=$run_id"
