#!/bin/bash
#SBATCH --job-name=ppg_r7_deep
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
mode=${1:?mode}; router=${2:?router directory}; seed=${3:-20260821}; work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; run=event120-v1_round7_${mode}_seed${seed}_job${SLURM_JOB_ID}; out=$work/outputs/$run
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
common=(--train-features $work/outputs/event120-v1_phase6d_qgh_routing/oof/oof_risk_features.parquet --validation-features $work/outputs/event120-v1_phase6e_continuous/validation_features.parquet --routed-features $router/routed_features.parquet --output $out --seed $seed)
if [[ $mode == phenotype_hard || $mode == phenotype_soft ]]; then python -m pulsedb_fewshot.round7_deep train-experts "${common[@]}" --routing ${mode#phenotype_}; elif [[ $mode == embedding_gru ]]; then python -m pulsedb_fewshot.round7_deep train-embedding-gru "${common[@]}"; else echo "unsupported mode: $mode" >&2; exit 2; fi
mkdir -p $archive/outputs/$run; rsync -a $out/ $archive/outputs/$run/; diff -qr $out $archive/outputs/$run
