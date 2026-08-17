#!/bin/bash

set -euo pipefail

population_checkpoint="${1:?population checkpoint is required}"
seed="${2:-20260813}"
existing_huber_job="${3:-819}"
work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
state_root="$work_root/outputs/submission_manifests"
demographics="$work_root/data/manifests/phase6_demographics/participant_demographics_clean.parquet"
mkdir -p "$state_root" "$(dirname "$demographics")"
cd "$project_root"

python -m pulsedb_fewshot.prepare_demographics \
  --input "$work_root/data/manifests/pulsedb_v2_full_cohort/development_segments.parquet" \
  --output "$demographics"

bp_change_job=$(sbatch --parsable \
  --job-name=ppg_p6_bpchange \
  scripts/sbatch_phase6_candidate.sh \
  bp_change_sampling "$seed" "$population_checkpoint" rolling_recent mse 0.5 \
  bp_change_aware mean no no '' 2.0)

robust_anchor_job=$(sbatch --parsable \
  --job-name=ppg_p6_anchor \
  scripts/sbatch_phase6_candidate.sh \
  robust_median_anchor "$seed" "$population_checkpoint" rolling_recent mse 0.5 \
  participant_balanced median no no '' 2.0)

quality_gate_job=$(sbatch --parsable \
  --job-name=ppg_p6_qgate \
  scripts/sbatch_phase6_candidate.sh \
  ppg_quality_gate "$seed" "$population_checkpoint" rolling_recent mse 0.5 \
  participant_balanced mean yes no '' 2.0)

demographic_job=$(sbatch --parsable \
  --job-name=ppg_p6_demo \
  scripts/sbatch_phase6_candidate.sh \
  demographic_conditioning "$seed" "$population_checkpoint" rolling_recent mse 0.5 \
  participant_balanced mean no yes "$demographics" 2.0)

manifest="$state_root/event120-v1_phase6_isolated_routes_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "PHASE=single-seed development screening"
  echo "SEED=$seed"
  echo "POPULATION_CHECKPOINT=$population_checkpoint"
  echo "ROBUST_LOSS_JOB=$existing_huber_job"
  echo "BP_CHANGE_SAMPLING_JOB=$bp_change_job"
  echo "ROBUST_ANCHOR_JOB=$robust_anchor_job"
  echo "PPG_QUALITY_GATE_JOB=$quality_gate_job"
  echo "DEMOGRAPHIC_CONDITIONING_JOB=$demographic_job"
  echo "FIXED_FIRST_PROTOCOL_JOB=818"
  echo "MULTI_SEED_SUBMITTED=no"
  echo "MULTI_FOLD_SUBMITTED=no"
  echo "LOCKED_META_TEST_ACCESSED=no"
} | tee "$manifest"

echo "PHASE6_ISOLATED_ROUTES_SUBMITTED=yes"
