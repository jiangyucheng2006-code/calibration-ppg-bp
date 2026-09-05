#!/bin/bash
#SBATCH --job-name=ppg_personal_diag
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
ppg_work="/home/$USER/work/ppg_bp"
conda activate "$ppg_work/envs/train"
ppg_project="${PPG_PROJECT_ROOT:?immutable source snapshot required}"
test ! -w "$ppg_project"
export PYTHONPATH="$ppg_project/src"
export PYTHONDONTWRITEBYTECODE=1
ppg_run="${1:?source run required}"
ppg_name="personal-mechanism-v1_$(basename "$ppg_run")_job${SLURM_JOB_ID}"
python -m pulsedb_fewshot.personal_mechanism_diagnostic \
  --run "$ppg_run" --store-root "$ppg_work/data/processed/development-calbased-analogue-v1" \
  --output "$ppg_work/outputs/$ppg_name"
mkdir -p "/home/$USER/nas/ppg_bp/outputs/$ppg_name"
rsync -a "$ppg_work/outputs/$ppg_name/" "/home/$USER/nas/ppg_bp/outputs/$ppg_name/"
cmp "$ppg_work/outputs/$ppg_name/diagnostic.json" "/home/$USER/nas/ppg_bp/outputs/$ppg_name/diagnostic.json"
