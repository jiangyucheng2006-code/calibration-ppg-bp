#!/bin/bash

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
state_root="$work_root/outputs/submission_manifests"
mkdir -p "$state_root"
cd "$project_root"

materialize_job="${1:?materialization job ID is required}"
torch_job="${2:?PyTorch installation job ID is required}"

smoke_job=$(sbatch --parsable \
  --dependency="afterok:${materialize_job}:${torch_job}" \
  scripts/sbatch_gpu_smoke.sh)

population_job=$(sbatch --parsable \
  --dependency="afterok:${smoke_job}" \
  --job-name=ppg_population \
  scripts/sbatch_train_method.sh population 20260813)

population_checkpoint="$work_root/outputs/event120-v1_population_seed20260813_job${population_job}/best.pt"
siamese_job=$(sbatch --parsable \
  --dependency="afterok:${population_job}" \
  --job-name=ppg_siamese \
  scripts/sbatch_train_method.sh siamese 20260813 "$population_checkpoint")
m0_job=$(sbatch --parsable \
  --dependency="afterok:${population_job}" \
  --job-name=ppg_m0 \
  scripts/sbatch_train_method.sh m0 20260813 "$population_checkpoint")
m1_job=$(sbatch --parsable \
  --dependency="afterok:${m0_job}" \
  --job-name=ppg_m1 \
  scripts/sbatch_train_method.sh m1 20260813 "$population_checkpoint")
m2_job=$(sbatch --parsable \
  --dependency="afterok:${m1_job}" \
  --job-name=ppg_m2 \
  scripts/sbatch_train_method.sh m2 20260813 "$population_checkpoint")

manifest="$state_root/event120-v1_suite_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "MATERIALIZE_JOB=$materialize_job"
  echo "TORCH_INSTALL_JOB=$torch_job"
  echo "GPU_SMOKE_JOB=$smoke_job"
  echo "POPULATION_JOB=$population_job"
  echo "SIAMESE_JOB=$siamese_job"
  echo "M0_JOB=$m0_job"
  echo "M1_JOB=$m1_job"
  echo "M2_JOB=$m2_job"
} | tee "$manifest"

echo "TRAINING_SUITE_SUBMITTED=yes"
