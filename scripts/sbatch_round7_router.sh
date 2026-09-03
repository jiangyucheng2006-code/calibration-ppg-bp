#!/bin/bash
#SBATCH --job-name=ppg_r7_router
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
seed=${1:-20260821}; work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; run=event120-v1_round7_phenotype_router_seed${seed}_job${SLURM_JOB_ID}; out=$work/outputs/$run
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
python -m pulsedb_fewshot.round7_deep train-router --train-features $work/outputs/event120-v1_phase6d_qgh_routing/oof/oof_risk_features.parquet --validation-features $work/outputs/event120-v1_phase6e_continuous/validation_features.parquet --embeddings $work/outputs/event120-v1_phase6e_continuous/waveform_embeddings.parquet --output $out --seed $seed --clusters 8
mkdir -p $archive/outputs/$run; rsync -a $out/ $archive/outputs/$run/; diff -qr $out $archive/outputs/$run
