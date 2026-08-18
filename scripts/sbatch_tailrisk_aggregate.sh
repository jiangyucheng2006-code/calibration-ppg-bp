#!/bin/bash
#SBATCH --job-name=ppg_tail_oof
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

folds_path="${1:?fold table is required}"
shift
if [[ "$#" -ne 5 ]]; then
  echo "ERROR: exactly five cross-fit M0 run directories are required" >&2
  exit 1
fi

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
output="$work_root/outputs/event120-v1_tailrisk-v1/oof"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

command=(
  python -m pulsedb_fewshot.tail_risk aggregate-oof
  --store-root "$work_root/data/processed/event120-v1"
  --folds "$folds_path"
  --output-dir "$output"
  --tail-fraction 0.30
)
for run_dir in "$@"; do
  test -f "$run_dir/run.json"
  test -f "$run_dir/best_validation_predictions.parquet"
  command+=(--fold-run "$run_dir")
done
"${command[@]}"

mkdir -p "$archive_root/outputs/event120-v1_tailrisk-v1/oof"
rsync -a "$output/" "$archive_root/outputs/event120-v1_tailrisk-v1/oof/"
diff -qr "$output" "$archive_root/outputs/event120-v1_tailrisk-v1/oof"
echo "TAILRISK_OOF_AGGREGATION_COMPLETE=yes"
