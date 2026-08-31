#!/usr/bin/env bash
#SBATCH --job-name=ss-component-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/%x_%j.out
#SBATCH --error=/home/jiangyu.cheng/work/ppg_bp/logs/%x_%j.err

set -euo pipefail

: "${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must identify the immutable source snapshot}"
: "${PPG_SMOKE_OUTPUT:?PPG_SMOKE_OUTPUT must identify the JSON result path}"

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jiangyu.cheng/work/ppg_bp/envs/train
export PYTHONPATH="$PPG_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$PPG_PROJECT_ROOT"

python -m pulsedb_fewshot.same_subject_component_smoke \
  --output "$PPG_SMOKE_OUTPUT"
