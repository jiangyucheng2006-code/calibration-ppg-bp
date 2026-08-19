#!/bin/bash
#SBATCH --job-name=ppg_p6e_train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
method=${1:?method}; seed=${2:-20260820}; work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; run=event120-v1_phase6e_${method}_seed${seed}_job${SLURM_JOB_ID}; out=$work/outputs/$run
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
python -m pulsedb_fewshot.phase6e_residual train --method $method --train-features $work/outputs/event120-v1_phase6d_qgh_routing/oof/oof_risk_features.parquet --validation-features $work/outputs/event120-v1_phase6e_continuous/validation_features.parquet --output $out --seed $seed
mkdir -p $archive/outputs/$run; rsync -a $out/ $archive/outputs/$run/; diff -qr $out $archive/outputs/$run
