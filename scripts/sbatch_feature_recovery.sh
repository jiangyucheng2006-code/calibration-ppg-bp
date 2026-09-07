#!/bin/bash
#SBATCH --job-name=ppg_feature_recovery
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
ppg_mode="${1:?smoke or train required}"
ppg_work="/home/$USER/work/ppg_bp"
ppg_nas="/home/$USER/nas/ppg_bp"
ppg_project="${PPG_PROJECT_ROOT:?original immutable snapshot required}"
ppg_recovery="${PPG_RECOVERY_ROOT:?recovery snapshot required}"
test ! -w "$ppg_project"
test ! -w "$ppg_recovery"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ppg_work/envs/train"
export PYTHONPATH="$ppg_project/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ppg_project"
if [[ "$ppg_mode" == smoke ]]; then
  ppg_name="personal-feature-recovery-smoke_job${SLURM_JOB_ID}"
  python "$ppg_recovery/scripts/check_feature_recovery.py" \
    --store-root "$ppg_work/data/processed/development-calbased-analogue-v1" \
    --output "$ppg_work/outputs/$ppg_name"
elif [[ "$ppg_mode" == train ]]; then
  ppg_name="personal-feature-mechanisms-v1_random_disjoint_shared_bilinear64_seed20260906_job${SLURM_JOB_ID}"
  # Exact original model/data/optimization, restarted from epoch zero.
  # workers=0 avoids the multiprocessing shared-memory collation path.
  python -m pulsedb_fewshot.personal_feature_train --candidate shared_bilinear64 \
    --split-mode random_disjoint --seed 20260906 --epochs 0 --patience 8 \
    --batch-size 64 --workers 0 --examples-per-epoch 200000 --require-cuda \
    --store-root "$ppg_work/data/processed/development-calbased-analogue-v1" \
    --output "$ppg_work/outputs/$ppg_name"
else
  echo 'invalid mode' >&2
  exit 2
fi
mkdir -p "$ppg_nas/outputs/$ppg_name"
rsync -a "$ppg_work/outputs/$ppg_name/" "$ppg_nas/outputs/$ppg_name/"
if [[ "$ppg_mode" == train ]]; then
  mkdir -p "$ppg_nas/checkpoints/$ppg_name"
  rsync -a "$ppg_work/outputs/$ppg_name/best.pt" "$ppg_nas/checkpoints/$ppg_name/"
  cmp "$ppg_work/outputs/$ppg_name/run.json" "$ppg_nas/outputs/$ppg_name/run.json"
  cmp "$ppg_work/outputs/$ppg_name/participant_profile_index.parquet" "$ppg_nas/outputs/$ppg_name/participant_profile_index.parquet"
else
  cmp "$ppg_work/outputs/$ppg_name/smoke.json" "$ppg_nas/outputs/$ppg_name/smoke.json"
fi
