#!/bin/bash
#SBATCH --job-name=ppg_p6d_risk
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

seed="${1:-20260819}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
pipeline_root="$work_root/outputs/event120-v1_phase6d_qgh_routing"
features="$pipeline_root/oof/oof_risk_features.parquet"
output="$pipeline_root/risk_classifier"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"
test -f "$features"

python -m pulsedb_fewshot.tail_risk train-risk \
  --features "$features" \
  --output-dir "$output" \
  --validation-fold 4 \
  --seed "$seed" \
  --epochs 100 \
  --patience 10 \
  --batch-size 512

mkdir -p "$archive_root/outputs/event120-v1_phase6d_qgh_routing/risk_classifier"
rsync -a "$output/" "$archive_root/outputs/event120-v1_phase6d_qgh_routing/risk_classifier/"
diff -qr "$output" "$archive_root/outputs/event120-v1_phase6d_qgh_routing/risk_classifier"
echo "PHASE6D_RISK_COMPLETE=yes"
