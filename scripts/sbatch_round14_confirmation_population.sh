#!/bin/bash
#SBATCH --job-name=ppg_r14_pop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

backbone="${1:?backbone is required}"
seed="${2:?seed is required}"
case "$backbone" in
  resnet_small|inception_time_wide) ;;
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
run_id="event120-v1_round14_confirmation_${backbone}_population_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

test -d "$project_root"
test ! -w "$project_root"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.train \
  --method population \
  --backbone "$backbone" \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$output" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --workers 4 \
  --episodes-per-epoch 99968 \
  --feature-dim 256 \
  --crossfit-folds "$folds" \
  --crossfit-fit-folds 0 1 2 \
  --crossfit-validation-fold 3 \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
diff -qr "$output" "$archive_root/outputs/$run_id" >/dev/null
echo "ROUND14_CONFIRMATION_POPULATION_COMPLETE=$run_id"
