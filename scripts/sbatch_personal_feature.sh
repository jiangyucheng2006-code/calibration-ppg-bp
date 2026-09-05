#!/bin/bash
#SBATCH --job-name=ppg_feature
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
candidate="${1:?candidate required}"
split_mode="${2:?split required}"
seed="${3:-20260906}"
ppg_work="/home/$USER/work/ppg_bp"
ppg_nas="/home/$USER/nas/ppg_bp"
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ppg_work/envs/train"
export PYTHONPATH="$ppg_project/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$ppg_project"
ppg_name="personal-feature-mechanisms-v1_${split_mode}_${candidate}_seed${seed}_job${SLURM_JOB_ID}"
python -m pulsedb_fewshot.personal_feature_train --candidate "$candidate" \
  --split-mode "$split_mode" --seed "$seed" --epochs 0 --patience 8 \
  --batch-size 64 --workers 4 --examples-per-epoch 200000 --require-cuda \
  --store-root "$ppg_work/data/processed/development-calbased-analogue-v1" \
  --output "$ppg_work/outputs/$ppg_name"
mkdir -p "$ppg_nas/outputs/$ppg_name" "$ppg_nas/checkpoints/$ppg_name"
rsync -a "$ppg_work/outputs/$ppg_name/" "$ppg_nas/outputs/$ppg_name/"
rsync -a "$ppg_work/outputs/$ppg_name/best.pt" "$ppg_nas/checkpoints/$ppg_name/"
cmp "$ppg_work/outputs/$ppg_name/run.json" "$ppg_nas/outputs/$ppg_name/run.json"
cmp "$ppg_work/outputs/$ppg_name/participant_profile_index.parquet" "$ppg_nas/outputs/$ppg_name/participant_profile_index.parquet"
