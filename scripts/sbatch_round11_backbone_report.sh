#!/bin/bash
#SBATCH --job-name=ppg_r11_report
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
run_id="event120-v1_round11a_backbone_report_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

arguments=()
for specification in "$@"; do
  arguments+=(--run "$specification")
done
python -m pulsedb_fewshot.round11_backbones report \
  "${arguments[@]}" \
  --reference-backbone resnet_small \
  --expected-seed "$seed" \
  --output "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "ROUND11A_REPORT_COMPLETE=$run_id"
