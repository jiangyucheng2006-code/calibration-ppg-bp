#!/bin/bash

set -euo pipefail

population_checkpoint="${1:?population checkpoint is required}"
seed="${2:-20260813}"
work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
state_root="$work_root/outputs/submission_manifests"
mkdir -p "$state_root"
cd "$project_root"

fixed_first_job=$(sbatch --parsable \
  --job-name=ppg_p6_fixed \
  scripts/sbatch_phase6_candidate.sh \
  fixed_first "$seed" "$population_checkpoint" fixed_first mse 0.5)

robust_loss_job=$(sbatch --parsable \
  --job-name=ppg_p6_huber \
  scripts/sbatch_phase6_candidate.sh \
  huber_rolling "$seed" "$population_checkpoint" rolling_recent huber 0.5)

manifest="$state_root/event120-v1_phase6_single_seed_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "PHASE=single-seed development screening"
  echo "SEED=$seed"
  echo "POPULATION_CHECKPOINT=$population_checkpoint"
  echo "FIXED_FIRST_JOB=$fixed_first_job"
  echo "ROBUST_LOSS_JOB=$robust_loss_job"
  echo "MULTI_SEED_SUBMITTED=no"
  echo "MULTI_FOLD_SUBMITTED=no"
} | tee "$manifest"

echo "PHASE6_EXPLORATION_SUBMITTED=yes"
