#!/bin/bash

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
state_root="$work_root/outputs/submission_manifests"
run_prefix="event120-v1_repeat5-v1"
seeds=(20260813 20260814 20260815 20260816 20260817)
max_epochs=0
patience=8

mkdir -p "$state_root" "$archive_root/outputs/submission_manifests"
cd "$project_root"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
python -m pytest -q

timestamp="$(date +%Y%m%d-%H%M%S)"
manifest="$state_root/${run_prefix}_${timestamp}.tsv"
snapshot="$archive_root/code_snapshots/${run_prefix}_${timestamp}.tar.gz"
mkdir -p "$(dirname "$snapshot")"
tar \
  --exclude=.git \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  -czf "$snapshot" .
snapshot_sha256="$(sha256sum "$snapshot" | cut -d' ' -f1)"

printf 'role\tseed\tjob_id\tdependency\tcheckpoint\n' > "$manifest"
printf '# run_prefix=%s\n# max_epochs=%s\n# patience=%s\n# code_snapshot=%s\n# code_snapshot_sha256=%s\n' \
  "$run_prefix" "$max_epochs" "$patience" "$snapshot" "$snapshot_sha256" >> "$manifest"

smoke_job="$(sbatch --parsable --job-name=ppg_repeat_smoke scripts/sbatch_gpu_smoke.sh)"
smoke_job="${smoke_job%%;*}"
printf 'smoke\tNA\t%s\tnone\tNA\n' "$smoke_job" >> "$manifest"

all_jobs=()
for seed in "${seeds[@]}"; do
  suffix="${seed: -4}"
  population_job="$(sbatch --parsable \
    --dependency="afterok:${smoke_job}" \
    --job-name="ppg_pop_${suffix}" \
    scripts/sbatch_train_method.sh population "$seed" '' "$max_epochs" "$patience" "$run_prefix")"
  population_job="${population_job%%;*}"
  population_checkpoint="$work_root/outputs/${run_prefix}_population_seed${seed}_job${population_job}/best.pt"
  printf 'population\t%s\t%s\tafterok:%s\t%s\n' \
    "$seed" "$population_job" "$smoke_job" "$population_checkpoint" >> "$manifest"
  all_jobs+=("$population_job")

  for method in m0 m1 m2; do
    method_job="$(sbatch --parsable \
      --dependency="afterok:${population_job}" \
      --job-name="ppg_${method}_${suffix}" \
      scripts/sbatch_train_method.sh "$method" "$seed" "$population_checkpoint" \
      "$max_epochs" "$patience" "$run_prefix")"
    method_job="${method_job%%;*}"
    printf '%s\t%s\t%s\tafterok:%s\t%s\n' \
      "$method" "$seed" "$method_job" "$population_job" "$population_checkpoint" >> "$manifest"
    all_jobs+=("$method_job")
  done

  controls_job="$(sbatch --parsable \
    --dependency="afterok:${population_job}" \
    --job-name="ppg_ctl_${suffix}" \
    scripts/sbatch_evaluate_calibration.sh "$population_job" "$seed" \
    "$population_checkpoint" "$run_prefix")"
  controls_job="${controls_job%%;*}"
  printf 'calibration_controls\t%s\t%s\tafterok:%s\t%s\n' \
    "$seed" "$controls_job" "$population_job" "$population_checkpoint" >> "$manifest"
  all_jobs+=("$controls_job")
done

dependency="afterok:$(IFS=:; echo "${all_jobs[*]}")"
aggregate_job="$(sbatch --parsable \
  --dependency="$dependency" \
  scripts/sbatch_aggregate_repeat_seeds.sh "$run_prefix" "${seeds[@]}")"
aggregate_job="${aggregate_job%%;*}"
printf 'aggregate\tNA\t%s\t%s\tNA\n' "$aggregate_job" "$dependency" >> "$manifest"

cp "$manifest" "$archive_root/outputs/submission_manifests/$(basename "$manifest")"
cmp "$manifest" "$archive_root/outputs/submission_manifests/$(basename "$manifest")"

printf 'RUN_PREFIX=%s\n' "$run_prefix"
printf 'SEEDS=%s\n' "${seeds[*]}"
printf 'MAX_EPOCHS=%s\n' "$max_epochs"
printf 'PATIENCE=%s\n' "$patience"
printf 'SMOKE_JOB=%s\n' "$smoke_job"
printf 'AGGREGATE_JOB=%s\n' "$aggregate_job"
printf 'MANIFEST=%s\n' "$manifest"
printf 'CODE_SNAPSHOT=%s\n' "$snapshot"
echo "REPEAT_SEED_SUITE_SUBMITTED=yes"
