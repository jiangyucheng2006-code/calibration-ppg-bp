#!/bin/bash
#SBATCH --job-name=ppg_repeat_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/$USER/work/ppg_bp/envs/train

project=/home/$USER/work/ppg_bp/code/pulsedb_fewshot
run_root=/home/$USER/work/ppg_bp/outputs
archive_root=/home/$USER/nas/ppg_bp/outputs
run_prefix=event120-v1_repeat5-v1
output=$run_root/${run_prefix}_report_job${SLURM_JOB_ID}

cd "$project"
python -m pulsedb_fewshot.report_repeat_seeds \
  --run-root "$run_root" \
  --run-prefix "$run_prefix" \
  --seeds 20260813,20260814,20260815,20260816,20260817 \
  --output-dir "$output"

mkdir -p "$archive_root"
rsync -a "$output/" "$archive_root/$(basename "$output")/"
diff -qr "$output" "$archive_root/$(basename "$output")"

echo "REPEAT_SEED_REPORT_ARCHIVED=yes"
echo "WORK_OUTPUT=$output"
echo "NAS_OUTPUT=$archive_root/$(basename "$output")"
