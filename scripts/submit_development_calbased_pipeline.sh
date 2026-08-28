#!/bin/bash

set -euo pipefail

seed="${1:-20260828}"
work_root="/home/$USER/work/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
store_root="$work_root/data/processed/development-calbased-analogue-v1"

test -d "$project_root"
test ! -w "$project_root"
cd "$project_root"

prep_job=""
if [[ -f "$store_root/materialization.json" ]]; then
  echo "DATA_STORE_ALREADY_PRESENT=$store_root"
else
  test ! -e "$store_root" || {
    echo "ERROR: partial/non-audited store exists and must be inspected: $store_root" >&2
    exit 1
  }
  prep_job="$(sbatch --parsable \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    scripts/sbatch_development_calbased_prepare.sh "$seed")"
  prep_job="${prep_job%%;*}"
  export PPG_PREP_DEPENDENCY="$prep_job"
  echo "DEVELOPMENT_CALBASED_PREP_JOB=$prep_job"
fi

bash scripts/submit_development_calbased_screen.sh both "$seed"

if [[ -n "$prep_job" ]]; then
  echo "All model jobs are queued after successful data preparation job $prep_job."
else
  echo "All model jobs were submitted against the existing audited store."
fi
