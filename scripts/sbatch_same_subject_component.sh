#!/bin/bash
#SBATCH --job-name=ppg_ss_component
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

candidate="${1:?candidate is required}"
seed="${2:-20260831}"
batch_size="${3:-128}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
store_root="$work_root/data/processed/development-calbased-analogue-v1"
demographics="$work_root/data/manifests/phase6_demographics/participant_demographics_clean.parquet"
beat_similarity="${PPG_BEAT_SIMILARITY_PATH:-}"
run_id="same-subject-single-component-v1_random_disjoint_${candidate}_seed${seed}_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
test -d "$project_root"
test ! -w "$project_root"
test -f "$demographics"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

arguments=(
  --candidate "$candidate"
  --store-root "$store_root"
  --split-mode random_disjoint
  --output "$output"
  --demographics-path "$demographics"
  --seed "$seed"
  --epochs 0
  --patience 8
  --batch-size "$batch_size"
  --workers 4
  --examples-per-epoch 200000
  --require-cuda
)
if [[ -n "$beat_similarity" ]]; then
  arguments+=(--beat-similarity-path "$beat_similarity")
fi
python -m pulsedb_fewshot.same_subject_component_train "${arguments[@]}"

mkdir -p "$archive_root/outputs/$run_id" "$archive_root/checkpoints/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
rsync -a "$output/best.pt" "$archive_root/checkpoints/$run_id/best.pt"
cmp -s "$output/run.json" "$archive_root/outputs/$run_id/run.json"
echo "SAME_SUBJECT_COMPONENT_COMPLETE=$run_id"

