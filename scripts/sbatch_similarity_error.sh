#!/bin/bash
#SBATCH --job-name=ppg_similarity_error
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail

work=/home/$USER/work/ppg_bp
archive=/home/$USER/nas/ppg_bp
root=$work/code/pulsedb_fewshot
run=event120-v1_similarity_error_job${SLURM_JOB_ID}
output=$work/outputs/$run

source /opt/conda/etc/profile.d/conda.sh
conda activate $work/envs/train
cd $root

python -m pulsedb_fewshot.analyze_similarity_error \
  --similarity "$work/outputs/event120-v1_meta_validation_10s_beat_similarity_job946/window_similarity_private.parquet" \
  --run "General QGate + Huber=$work/outputs/event120-v1_phase6b_qgate_huber_seed20260813_job841/best_validation_predictions.parquet" \
  --run "R7-5 Causal GRU=$work/outputs/event120-v1_phase6e_causal_gru_seed20260821_job936/predictions.parquet" \
  --output "$output" \
  --bootstrap-repetitions 5000 \
  --seed 20260820

mkdir -p "$archive/outputs/$run"
rsync -a "$output/" "$archive/outputs/$run/"
diff -qr "$output" "$archive/outputs/$run"
