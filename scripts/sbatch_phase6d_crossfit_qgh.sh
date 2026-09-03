#!/bin/bash
#SBATCH --job-name=ppg_p6d_xqgh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

fold="${1:?held-out fold is required}"
folds_path="${2:?fold table is required}"
seed="${3:?seed is required}"
population_checkpoint="${4:?fold-specific population checkpoint is required}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_id="event120-v1_phase6d_xfit_qgate_huber_fold${fold}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"
test -f "$folds_path"
test -f "$population_checkpoint"

python -m pulsedb_fewshot.train \
  --method m0 \
  --store-root "$work_root/data/processed/event120-v1" \
  --output "$output" \
  --population-checkpoint "$population_checkpoint" \
  --seed "$seed" \
  --epochs 0 \
  --patience 8 \
  --batch-size 64 \
  --workers 3 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --episodes-per-epoch 100000 \
  --train-support-policy fixed_first \
  --loss huber \
  --huber-delta 0.5 \
  --episode-sampling participant_balanced \
  --anchor-mode mean \
  --use-quality-gate \
  --ks 5 \
  --crossfit-folds "$folds_path" \
  --crossfit-heldout-fold "$fold" \
  --require-cuda

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "PHASE6D_CROSSFIT_COMPLETE=$run_id"
