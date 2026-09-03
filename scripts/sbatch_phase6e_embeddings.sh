#!/bin/bash
#SBATCH --job-name=ppg_p6e_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; out=$work/outputs/event120-v1_phase6e_continuous/waveform_embeddings.parquet
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
python -m pulsedb_fewshot.phase6e_residual extract-embeddings --store-root $work/data/processed/event120-v1 --population-checkpoint $work/outputs/event120-v1_repeat5-v1_population_seed20260813_job782/best.pt --train-features $work/outputs/event120-v1_phase6d_qgh_routing/oof/oof_risk_features.parquet --validation-features $work/outputs/event120-v1_phase6e_continuous/validation_features.parquet --output $out
mkdir -p $archive/outputs/event120-v1_phase6e_continuous; cp $out ${out%.parquet}.json $archive/outputs/event120-v1_phase6e_continuous/
