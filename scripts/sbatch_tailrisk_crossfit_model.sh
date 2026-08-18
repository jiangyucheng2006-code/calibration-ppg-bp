#!/bin/bash
#SBATCH --job-name=ppg_tail_xfit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

stage="${1:?stage is required: population or m0}"
fold="${2:?held-out fold is required}"
folds_path="${3:?fold table is required}"
seed="${4:?seed is required}"
population_checkpoint="${5:-}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"

case "$stage" in
  population)
    method="population"
    batch_size=256
    episodes=200000
    ;;
  m0)
    method="m0"
    batch_size=64
    episodes=100000
    test -f "$population_checkpoint"
    ;;
  *)
    echo "ERROR: stage must be population or m0" >&2
    exit 1
    ;;
esac

run_id="event120-v1_tailrisk_xfit_${stage}_fold${fold}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

command=(
  python -m pulsedb_fewshot.train
  --method "$method"
  --store-root "$work_root/data/processed/event120-v1"
  --output "$output"
  --seed "$seed"
  --epochs 0
  --patience 8
  --batch-size "$batch_size"
  --workers 3
  --learning-rate 0.0003
  --weight-decay 0.0001
  --episodes-per-epoch "$episodes"
  --train-support-policy fixed_first
  --crossfit-folds "$folds_path"
  --crossfit-heldout-fold "$fold"
  --require-cuda
)
if [[ "$stage" == "m0" ]]; then
  command+=(--population-checkpoint "$population_checkpoint" --ks 5)
fi
"${command[@]}"

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
diff -qr "$output" "$archive_root/outputs/$run_id"
echo "TAILRISK_CROSSFIT_COMPLETE=$run_id"
