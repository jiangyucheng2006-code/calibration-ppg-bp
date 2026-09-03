#!/bin/bash

set -euo pipefail

population_checkpoint="${1:?population checkpoint is required}"
seed="${2:-20260813}"
work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
state_root="$work_root/outputs/submission_manifests"
mkdir -p "$state_root"
cd "$project_root"

test -f "$population_checkpoint" || {
  echo "ERROR: population checkpoint is missing: $population_checkpoint" >&2
  exit 1
}

# Together with existing jobs 818 (none), 826 (Huber) and 829 (quality gate),
# these five jobs complete the 2x2x2 ablation of quality gate, Huber loss and
# participant-tail CVaR.  Every new job changes one component relative to an
# already available factorial neighbour.
cvar_job=$(sbatch --parsable --job-name=ppg_p6b_cvar \
  scripts/sbatch_phase6b_tail_candidate.sh \
  cvar "$seed" "$population_checkpoint" mse no mean_cvar)

qgate_huber_job=$(sbatch --parsable --job-name=ppg_p6b_qh \
  scripts/sbatch_phase6b_tail_candidate.sh \
  qgate_huber "$seed" "$population_checkpoint" huber yes mean)

qgate_cvar_job=$(sbatch --parsable --job-name=ppg_p6b_qc \
  scripts/sbatch_phase6b_tail_candidate.sh \
  qgate_cvar "$seed" "$population_checkpoint" mse yes mean_cvar)

huber_cvar_job=$(sbatch --parsable --job-name=ppg_p6b_hc \
  scripts/sbatch_phase6b_tail_candidate.sh \
  huber_cvar "$seed" "$population_checkpoint" huber no mean_cvar)

qgate_huber_cvar_job=$(sbatch --parsable --job-name=ppg_p6b_qhc \
  scripts/sbatch_phase6b_tail_candidate.sh \
  qgate_huber_cvar "$seed" "$population_checkpoint" huber yes mean_cvar)

manifest="$state_root/event120-v1_phase6b_tail_factorial_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "PHASE=Phase 6b single-seed tail-aware development screening"
  echo "REFERENCE_JOB=818"
  echo "EXISTING_HUBER_JOB=826"
  echo "EXISTING_QUALITY_GATE_JOB=829"
  echo "SEED=$seed"
  echo "TRAIN_SUPPORT_POLICY=fixed_first"
  echo "VALIDATION_SUPPORT_POLICY=fixed_first"
  echo "QUERY_POLICY=event_6_and_later"
  echo "TAIL_FRACTION=0.30"
  echo "TAIL_WEIGHT=0.50"
  echo "EARLY_STOPPING=overall_participant_macro_mae_patience_8"
  echo "CVAR_JOB=$cvar_job"
  echo "QUALITY_GATE_HUBER_JOB=$qgate_huber_job"
  echo "QUALITY_GATE_CVAR_JOB=$qgate_cvar_job"
  echo "HUBER_CVAR_JOB=$huber_cvar_job"
  echo "QUALITY_GATE_HUBER_CVAR_JOB=$qgate_huber_cvar_job"
  echo "MULTI_SEED_SUBMITTED=no"
  echo "LOCKED_META_TEST_ACCESSED=no"
} | tee "$manifest"

echo "PHASE6B_TAIL_FACTORIAL_SUBMITTED=yes"
