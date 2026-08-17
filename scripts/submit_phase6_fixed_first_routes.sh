#!/bin/bash

set -euo pipefail

population_checkpoint="${1:?population checkpoint is required}"
seed="${2:-20260813}"
work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
state_root="$work_root/outputs/submission_manifests"
demographics="$work_root/data/manifests/phase6_demographics/participant_demographics_clean.parquet"
mkdir -p "$state_root"
cd "$project_root"

test -f "$demographics" || {
  echo "ERROR: demographic feature table is missing: $demographics" >&2
  exit 1
}

robust_loss_job=$(sbatch --parsable \
  --job-name=ppg_p6_ff_huber \
  scripts/sbatch_phase6_candidate.sh \
  fixedfirst_huber "$seed" "$population_checkpoint" fixed_first huber 0.5 \
  participant_balanced mean no no '' 2.0)

bp_change_job=$(sbatch --parsable \
  --job-name=ppg_p6_ff_bpchange \
  scripts/sbatch_phase6_candidate.sh \
  fixedfirst_bp_change "$seed" "$population_checkpoint" fixed_first mse 0.5 \
  bp_change_aware mean no no '' 2.0)

robust_anchor_job=$(sbatch --parsable \
  --job-name=ppg_p6_ff_anchor \
  scripts/sbatch_phase6_candidate.sh \
  fixedfirst_median_anchor "$seed" "$population_checkpoint" fixed_first mse 0.5 \
  participant_balanced median no no '' 2.0)

quality_gate_job=$(sbatch --parsable \
  --job-name=ppg_p6_ff_qgate \
  scripts/sbatch_phase6_candidate.sh \
  fixedfirst_ppg_quality_gate "$seed" "$population_checkpoint" fixed_first mse 0.5 \
  participant_balanced mean yes no '' 2.0)

demographic_job=$(sbatch --parsable \
  --job-name=ppg_p6_ff_demo \
  scripts/sbatch_phase6_candidate.sh \
  fixedfirst_demographics "$seed" "$population_checkpoint" fixed_first mse 0.5 \
  participant_balanced mean no yes "$demographics" 2.0)

manifest="$state_root/event120-v1_phase6_fixed_first_routes_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "PHASE=single-seed fixed-first development screening"
  echo "REFERENCE_JOB=818"
  echo "SEED=$seed"
  echo "TRAIN_SUPPORT_POLICY=fixed_first"
  echo "VALIDATION_SUPPORT_POLICY=fixed_first"
  echo "QUERY_POLICY=event_6_and_later"
  echo "ROBUST_LOSS_JOB=$robust_loss_job"
  echo "BP_CHANGE_SAMPLING_JOB=$bp_change_job"
  echo "ROBUST_ANCHOR_JOB=$robust_anchor_job"
  echo "PPG_QUALITY_GATE_JOB=$quality_gate_job"
  echo "DEMOGRAPHIC_CONDITIONING_JOB=$demographic_job"
  echo "MULTI_SEED_SUBMITTED=no"
  echo "MULTI_FOLD_SUBMITTED=no"
  echo "LOCKED_META_TEST_ACCESSED=no"
} | tee "$manifest"

echo "PHASE6_FIXED_FIRST_ROUTES_SUBMITTED=yes"
