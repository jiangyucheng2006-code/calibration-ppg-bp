#!/bin/bash
#SBATCH --job-name=ppg_cb_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

seed="${1:-20260828}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
segment_index="$work_root/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet"
subject_splits="$work_root/data/manifests/pulsedb_v2_full_cohort/subject_splits.csv"
protocol_root="$work_root/data/manifests/development-calbased-analogue-v1_seed${seed}"
store_root="$work_root/data/processed/development-calbased-analogue-v1"
archive_protocol="$archive_root/data/manifests/development-calbased-analogue-v1_seed${seed}"
archive_store_manifest="$archive_root/data/manifests/development-calbased-analogue-v1_materialization.json"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
test -d "$project_root"
test ! -w "$project_root"
test -f "$segment_index"
test -f "$subject_splits"
test ! -e "$protocol_root"
test ! -e "$store_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"

python -m pulsedb_fewshot.calbased_prepare \
  --segment-index "$segment_index" \
  --subject-splits "$subject_splits" \
  --protocol-output "$protocol_root" \
  --store-output "$store_root" \
  --split-mode random_disjoint \
  --split-mode chronological_blocked \
  --seed "$seed" \
  --expected-subjects 2058 \
  --workers 4

test -f "$protocol_root/data_preparation.json"
test -f "$store_root/materialization.json"
mkdir -p "$(dirname "$archive_protocol")" "$(dirname "$archive_store_manifest")"
rsync -a "$protocol_root/" "$archive_protocol/"
cp -p "$store_root/materialization.json" "$archive_store_manifest"
cmp -s "$protocol_root/data_preparation.json" "$archive_protocol/data_preparation.json"
cmp -s "$store_root/materialization.json" "$archive_store_manifest"
echo "DEVELOPMENT_CALBASED_PREPARATION_COMPLETE=$store_root"
