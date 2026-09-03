#!/bin/bash

set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 POPULATION_RUN QGH_RUN EVALUATION_RUN SEED BASE_DEPENDENCY [GPU_A] [GPU_B]" >&2
  exit 2
fi

population_run="$1"
qgh_run="$2"
evaluation_run="$3"
seed="$4"
test "$seed" -eq 20260828
base_dependency="${5%%;*}"
gpu_a="${6:-rtx_5080}"
gpu_b="${7:-rtx_5070_ti}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable Round-14 snapshot}"
project_root="$(readlink -f "$project_root")"
scripts="$project_root/scripts"
manifest_root="$work_root/outputs/submission_manifests"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest="$manifest_root/round14_exploratory_seed${seed}_${timestamp}.tsv"

test -d "$project_root"
test ! -w "$project_root"
case "$(basename "$evaluation_run")" in
  *_job[0-9]*) evaluation_job="${evaluation_run##*_job}" ;;
  *) echo "EVALUATION_RUN must end in _job<SlurmJobID>" >&2; exit 2 ;;
esac
evaluation_job="${evaluation_job%%;*}"
[[ "$evaluation_job" =~ ^[0-9]+$ ]] || {
  echo "could not derive the evaluation job ID" >&2
  exit 2
}
prepare_dependency=()
if [[ "$base_dependency" != "none" ]]; then
  [[ "$base_dependency" =~ ^[0-9]+([:][0-9]+)*$ ]] || {
    echo "BASE_DEPENDENCY must be none or colon-separated Slurm job IDs" >&2
    exit 2
  }
  prepare_dependency=(--dependency="afterok:${base_dependency}")
else
  for path in "$population_run" "$qgh_run"; do
    [[ -f "$path/run.json" && -f "$path/best.pt" ]] || {
      echo "base artifacts are absent and BASE_DEPENDENCY=none: $path" >&2
      exit 1
    }
  done
fi
mkdir -p "$manifest_root" "$archive_root/outputs/submission_manifests"

prepare_job="$(sbatch --parsable \
  "${prepare_dependency[@]}" \
  --gres="gpu:${gpu_a}:1" \
  "$scripts/sbatch_round14_exploratory_prepare.sh" \
  "$population_run" "$qgh_run" "$seed")"
prepare_job="${prepare_job%%;*}"
cache_dir="$work_root/outputs/event120-v1_round14_exploratory_cache_seed${seed}_job${prepare_job}"

methods=(
  calibration_relative
  calibration_relative_standards
)
jobs=()
runs=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  if (( index % 2 == 0 )); then gpu="$gpu_a"; else gpu="$gpu_b"; fi
  job="$(sbatch --parsable \
    --dependency="afterok:${prepare_job}" \
    --gres="gpu:${gpu}:1" \
    "$scripts/sbatch_round14_exploratory_candidate.sh" \
    "$method" "$cache_dir" "$seed")"
  job="${job%%;*}"
  jobs+=("$job")
  runs+=("${method}=$work_root/outputs/event120-v1_round14_${method}_seed${seed}_job${job}")
done

dependency="$(IFS=:; echo "${jobs[*]}"):${evaluation_job}"
report_job="$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  "$scripts/sbatch_round14_exploratory_report.sh" \
  "$seed" "$evaluation_run" "${runs[@]}")"
report_job="${report_job%%;*}"

{
  printf 'role\tmethod\tjob_id\tdependency\tartifact\n'
  printf 'cache\tinception_time_wide\t%s\t%s\t%s\n' \
    "$prepare_job" "$base_dependency" "$cache_dir"
  for index in "${!methods[@]}"; do
    printf 'candidate\t%s\t%s\tafterok:%s\t%s\n' \
      "${methods[$index]}" "${jobs[$index]}" "$prepare_job" \
      "${runs[$index]#*=}"
  done
  printf 'report\tall\t%s\tafterok:%s\t%s\n' \
    "$report_job" "$dependency" \
    "$work_root/outputs/event120-v1_round14_exploratory_report_seed${seed}_job${report_job}"
} > "$manifest"
rsync -a --no-perms --no-owner --no-group "$manifest" \
  "$archive_root/outputs/submission_manifests/"

echo "ROUND14_EXPLORATORY_SUBMITTED=yes"
echo "MANIFEST=$manifest"
echo "PREPARE_JOB=$prepare_job"
echo "CANDIDATE_JOBS=${jobs[*]}"
echo "REPORT_JOB=$report_job"
