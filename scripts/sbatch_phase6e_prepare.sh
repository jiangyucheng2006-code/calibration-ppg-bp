#!/bin/bash
#SBATCH --job-name=ppg_p6e_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; out=$work/outputs/event120-v1_phase6e_continuous/validation_features.parquet
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
python -m pulsedb_fewshot.phase6e_residual prepare-validation --store-root $work/data/processed/event120-v1 --general-run $work/outputs/event120-v1_phase6b_qgate_huber_seed20260813_job841 --output $out
mkdir -p $archive/outputs/event120-v1_phase6e_continuous; cp $out ${out%.parquet}.json $archive/outputs/event120-v1_phase6e_continuous/
