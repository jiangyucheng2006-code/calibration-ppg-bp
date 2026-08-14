#!/bin/bash
#SBATCH --job-name=ppg_seed_summary
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

run_prefix="${1:?run prefix is required}"
shift
if [ "$#" -lt 3 ]; then
  echo "ERROR: at least three seeds are required" >&2
  exit 2
fi
case "$run_prefix" in
  ''|*[!A-Za-z0-9._-]*) echo "ERROR: run_prefix contains unsafe characters" >&2; exit 2 ;;
esac

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="${run_prefix}_aggregate_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"
seed_csv="$(IFS=,; echo "$*")"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -m pulsedb_fewshot.aggregate_seed_results \
  --run-root "$work_root/outputs" \
  --output "$output" \
  --run-prefix "$run_prefix" \
  --seeds "$seed_csv"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "REPEAT_SEED_AGGREGATION_COMPLETE=$run_id"
