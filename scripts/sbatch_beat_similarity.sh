#!/bin/bash
#SBATCH --job-name=ppg_beat_similarity
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
work=/home/$USER/work/ppg_bp
archive=/home/$USER/nas/ppg_bp
root=$work/code/pulsedb_fewshot
run=${PPG_BEAT_RUN:-event120-v1_meta_validation_10s_beat_similarity_job${SLURM_JOB_ID}}
out=$work/outputs/$run
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
arguments=(
  python -m pulsedb_fewshot.analyze_beat_similarity
  --store-root "$work/data/processed/event120-v1"
  --keys "$work/outputs/event120-v1_phase6e_continuous/validation_features.parquet"
  --output "$out"
  --sampling-rate 125
)
if [[ -n "${PPG_BEAT_LIMIT:-}" ]]; then
  arguments+=(--limit "$PPG_BEAT_LIMIT")
fi
"${arguments[@]}"
if [[ "${PPG_BEAT_ARCHIVE:-yes}" == "yes" ]]; then
  mkdir -p "$archive/outputs/$run"
  rsync -a "$out/" "$archive/outputs/$run/"
  diff -qr "$out" "$archive/outputs/$run"
fi
