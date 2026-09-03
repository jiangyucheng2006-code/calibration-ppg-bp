#!/bin/bash
#SBATCH --job-name=ppg_r14_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

backbone="${1:?backbone is required}"
population_run="${2:?population run directory is required}"
qgh_run="${3:?QGH run directory is required}"
seed="${4:?seed is required}"
batch_size="${5:?evaluation batch size is required}"
case "$backbone" in
  resnet_small) test "$batch_size" -eq 512 ;;
  inception_time_wide) test "$batch_size" -eq 256 ;;
  *) echo "ERROR: unsupported Round-14 backbone: $backbone" >&2; exit 2 ;;
esac
case "$seed" in
  20260828|20260829|20260830|20260831) ;;
  *) echo "ERROR: seed is outside the prespecified confirmation set: $seed" >&2; exit 2 ;;
esac

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name the immutable Round-14 snapshot}"
folds="$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet"
run_id="event120-v1_round14_confirmation_${backbone}_evaluation_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

test -d "$project_root"
test ! -w "$project_root"
test -f "$population_run/best.pt"
test -f "$qgh_run/best.pt"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.round11_backbones evaluate \
  --round-number 14 \
  --backbone "$backbone" \
  --store-root "$work_root/data/processed/event120-v1" \
  --folds "$folds" \
  --population-run "$population_run" \
  --qgh-run "$qgh_run" \
  --batch-size "$batch_size" \
  --output "$output"

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id" >/dev/null
echo "ROUND14_CONFIRMATION_EVALUATION_COMPLETE=$run_id"
