#!/bin/bash
#SBATCH --job-name=ppg_prs_smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
mode="${1:?split mode required}"
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work="/home/$USER/work/ppg_bp"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ppg_work/envs/train"
export PYTHONPATH="$ppg_project/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PPG_TEST_DEVICE=cuda
cd "$ppg_project"
python -m pytest -q -p no:cacheprovider tests/test_lora_prs.py
ppg_source="$ppg_work/outputs/same-subject-personal-profile-v1_${mode}_subject_lora_rank4_seed20260904_job1399"
if [[ "$mode" == chronological_blocked ]]; then
  ppg_source="$ppg_work/outputs/same-subject-personal-profile-v1_${mode}_subject_lora_rank4_seed20260904_job1408"
fi
ppg_name="lora-prs-continuation-v1_${mode}_smoke_job${SLURM_JOB_ID}"
python -m pulsedb_fewshot.lora_prs_smoke --split-mode "$mode" --source-run "$ppg_source" \
  --store-root "$ppg_work/data/processed/development-calbased-analogue-v1" --output "$ppg_work/outputs/$ppg_name"
mkdir -p "/home/$USER/nas/ppg_bp/outputs/$ppg_name"
rsync -a "$ppg_work/outputs/$ppg_name/" "/home/$USER/nas/ppg_bp/outputs/$ppg_name/"
