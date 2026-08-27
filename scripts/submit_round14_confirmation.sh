#!/bin/bash

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name the immutable Round-14 snapshot}"
manifest_root="$work_root/outputs/submission_manifests"
archive_manifest_root="$archive_root/outputs/submission_manifests"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest="$manifest_root/round14_paired_seed_confirmation_${timestamp}.tsv"
archive_manifest="$archive_manifest_root/$(basename "$manifest")"
seeds=(20260828 20260829 20260830 20260831)
backbones=(resnet_small inception_time_wide)
discovery_reference="${ROUND14_DISCOVERY_REFERENCE:?ROUND14_DISCOVERY_REFERENCE must name the verified Round-13 ResNet evaluation}"
discovery_candidate="${ROUND14_DISCOVERY_CANDIDATE:?ROUND14_DISCOVERY_CANDIDATE must name the verified Round-13 wide-Inception evaluation}"

test -d "$project_root"
test ! -w "$project_root"
test -f "$discovery_reference/run.json"
test -f "$discovery_candidate/run.json"
test -d "$archive_root/outputs/$(basename "$discovery_reference")"
test -d "$archive_root/outputs/$(basename "$discovery_candidate")"
diff -qr "$discovery_reference" "$archive_root/outputs/$(basename "$discovery_reference")" >/dev/null
diff -qr "$discovery_candidate" "$archive_root/outputs/$(basename "$discovery_candidate")" >/dev/null
case "$(basename "$discovery_reference")" in
  *_job[0-9]*) discovery_reference_job="${discovery_reference##*_job}" ;;
  *) echo "ERROR: discovery reference path must end in _job<SlurmJobID>" >&2; exit 2 ;;
esac
case "$(basename "$discovery_candidate")" in
  *_job[0-9]*) discovery_candidate_job="${discovery_candidate##*_job}" ;;
  *) echo "ERROR: discovery candidate path must end in _job<SlurmJobID>" >&2; exit 2 ;;
esac
discovery_reference_job="${discovery_reference_job%%;*}"
discovery_candidate_job="${discovery_candidate_job%%;*}"
[[ "$discovery_reference_job" =~ ^[0-9]+$ ]]
[[ "$discovery_candidate_job" =~ ^[0-9]+$ ]]
mkdir -p "$manifest_root" "$archive_manifest_root"
test ! -e "$manifest"
test ! -e "$archive_manifest"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"
python -m pytest -q -p no:cacheprovider
source_sha256="$(python -c 'from pathlib import Path; from pulsedb_fewshot.training import source_tree_sha256; print(source_tree_sha256(Path.cwd()))')"

printf 'stage\tbackbone\tseed\tjob_id\tdependency\trun\tgres\tmicrobatch\taccumulation\teffective_batch\n' > "$manifest"
printf 'existing_evaluation\tresnet_small\t20260827\t%s\tnone\t%s\texisting\t512\t1\t512\n' \
  "$discovery_reference_job" "$discovery_reference" >> "$manifest"
printf 'existing_evaluation\tinception_time_wide\t20260827\t%s\tnone\t%s\texisting\t256\t1\t256\n' \
  "$discovery_candidate_job" "$discovery_candidate" >> "$manifest"

smoke_5080="$(sbatch --parsable --gres=gpu:rtx_5080:1 \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  scripts/sbatch_round14_confirmation_smoke.sh)"
smoke_5080="${smoke_5080%%;*}"
smoke_5070="$(sbatch --parsable --gres=gpu:rtx_5070_ti:1 \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  scripts/sbatch_round14_confirmation_smoke.sh)"
smoke_5070="${smoke_5070%%;*}"
smoke_5080_run="$work_root/outputs/event120-v1_round14_confirmation_smoke_job${smoke_5080}"
smoke_5070_run="$work_root/outputs/event120-v1_round14_confirmation_smoke_job${smoke_5070}"
printf 'smoke\tall\tNA\t%s\tnone\t%s\tgpu:rtx_5080:1\tNA\tNA\tNA\n' \
  "$smoke_5080" "$smoke_5080_run" >> "$manifest"
printf 'smoke\tall\tNA\t%s\tnone\t%s\tgpu:rtx_5070_ti:1\tNA\tNA\tNA\n' \
  "$smoke_5070" "$smoke_5070_run" >> "$manifest"

reference_specs=("20260827=$discovery_reference")
candidate_specs=("20260827=$discovery_candidate")
evaluation_jobs=()
all_jobs=("$smoke_5080" "$smoke_5070")

for seed_index in "${!seeds[@]}"; do
  seed="${seeds[$seed_index]}"
  for backbone_index in "${!backbones[@]}"; do
    backbone="${backbones[$backbone_index]}"
    if (( (seed_index + backbone_index) % 2 == 0 )); then
      gres="gpu:rtx_5080:1"
      smoke_dependency="$smoke_5080"
    else
      gres="gpu:rtx_5070_ti:1"
      smoke_dependency="$smoke_5070"
    fi
    if [ "$backbone" = "resnet_small" ]; then
      evaluation_batch=512
    else
      evaluation_batch=256
    fi

    population_job="$(sbatch --parsable --dependency="afterok:${smoke_dependency}" \
      --gres="$gres" --export=ALL,PPG_PROJECT_ROOT="$project_root" \
      scripts/sbatch_round14_confirmation_population.sh "$backbone" "$seed")"
    population_job="${population_job%%;*}"
    population_run="$work_root/outputs/event120-v1_round14_confirmation_${backbone}_population_seed${seed}_job${population_job}"
    printf 'population\t%s\t%s\t%s\tafterok:%s\t%s\t%s\t32\t4\t128\n' \
      "$backbone" "$seed" "$population_job" "$smoke_dependency" \
      "$population_run" "$gres" >> "$manifest"
    all_jobs+=("$population_job")

    qgh_job="$(sbatch --parsable --dependency="afterok:${population_job}" \
      --gres="$gres" --export=ALL,PPG_PROJECT_ROOT="$project_root" \
      scripts/sbatch_round14_confirmation_qgh.sh \
      "$backbone" "$population_run" "$seed")"
    qgh_job="${qgh_job%%;*}"
    qgh_run="$work_root/outputs/event120-v1_round14_confirmation_${backbone}_qgh_seed${seed}_job${qgh_job}"
    printf 'qgh\t%s\t%s\t%s\tafterok:%s\t%s\t%s\t16\t4\t64\n' \
      "$backbone" "$seed" "$qgh_job" "$population_job" "$qgh_run" "$gres" \
      >> "$manifest"
    all_jobs+=("$qgh_job")

    evaluation_job="$(sbatch --parsable --dependency="afterok:${qgh_job}" \
      --gres="$gres" --export=ALL,PPG_PROJECT_ROOT="$project_root" \
      scripts/sbatch_round14_confirmation_evaluate.sh \
      "$backbone" "$population_run" "$qgh_run" "$seed" "$evaluation_batch")"
    evaluation_job="${evaluation_job%%;*}"
    evaluation_run="$work_root/outputs/event120-v1_round14_confirmation_${backbone}_evaluation_seed${seed}_job${evaluation_job}"
    printf 'evaluation\t%s\t%s\t%s\tafterok:%s\t%s\t%s\t%s\t1\t%s\n' \
      "$backbone" "$seed" "$evaluation_job" "$qgh_job" "$evaluation_run" \
      "$gres" "$evaluation_batch" "$evaluation_batch" >> "$manifest"
    evaluation_jobs+=("$evaluation_job")
    all_jobs+=("$evaluation_job")
    if [ "$backbone" = "resnet_small" ]; then
      reference_specs+=("${seed}=${evaluation_run}")
    else
      candidate_specs+=("${seed}=${evaluation_run}")
    fi
  done
done

dependency="$(IFS=:; echo "${evaluation_jobs[*]}")"
report_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  scripts/sbatch_round14_confirmation_report.sh \
  "${reference_specs[@]}" "${candidate_specs[@]}")"
report_job="${report_job%%;*}"
report_run="$work_root/outputs/event120-v1_round14_confirmation_report_job${report_job}"
printf 'report\tall\tall\t%s\tafterok:%s\t%s\tnone\tNA\tNA\tNA\n' \
  "$report_job" "$dependency" "$report_run" >> "$manifest"
all_jobs+=("$report_job")

archive_job="$(sbatch --parsable --dependency="afterok:${report_job}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  scripts/sbatch_round14_confirmation_archive.sh "$manifest")"
archive_job="${archive_job%%;*}"
archive_run="$archive_root/logs/round14_confirmation_job${archive_job}"
printf 'archive\tall\tall\t%s\tafterok:%s\t%s\tnone\tNA\tNA\tNA\n' \
  "$archive_job" "$report_job" "$archive_run" >> "$manifest"
all_jobs+=("$archive_job")

cp -- "$manifest" "$archive_manifest"
cmp -s "$manifest" "$archive_manifest"
printf 'ROUND14_SOURCE_TREE_SHA256=%s\n' "$source_sha256"
printf 'ROUND14_JOB_IDS=%s\n' "$(IFS=,; echo "${all_jobs[*]}")"
printf 'ROUND14_REPORT_JOB=%s\n' "$report_job"
printf 'ROUND14_ARCHIVE_JOB=%s\n' "$archive_job"
printf 'ROUND14_REPORT_RUN=%s\n' "$report_run"
printf 'ROUND14_SUBMISSION_MANIFEST=%s\n' "$manifest"
echo "ROUND14_CONFIRMATION_SUBMITTED=yes"
