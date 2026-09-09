#!/bin/bash
# Submit only after this exact immutable snapshot's smoke has succeeded.
set -euo pipefail
umask 077
[[ "$(hostname)" == slurm ]]
ppg_code="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_code"
ppg_tag="${PPG_BATCH_TAG:?batch tag required}"
[[ "$ppg_tag" =~ ^[0-9]{8}-[0-9]{6}$ ]]
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_root="$ppg_work/outputs/post-enrollment-30-v1_$ppg_tag"
ppg_store="${PPG_OFFICIAL_STORE:?source store required}"
ppg_smoke="${PPG_SMOKE_JOB:?completed smoke job required}"
[[ "$ppg_smoke" =~ ^[0-9]+$ ]]
[[ "$(sacct -n -X -j "$ppg_smoke" --format=State --parsable2 | tr -d ' ')" == COMPLETED ]]
[[ "$(sacct -n -X -j "$ppg_smoke" --format=ExitCode --parsable2 | tr -d ' ')" == 0:0 ]]
test -d "$ppg_root/smoke"
test ! -e "$ppg_root/submission.tsv"
ppg_gres="$(sinfo -N -h -n hpc-2 -o %G)"
[[ "$ppg_gres" == *gpu:rtx_5080:1* && "$ppg_gres" == *gpu:rtx_5070_ti:1* ]]
ppg_manifest="$ppg_root/submission.tsv"
printf 'stage\tjob_id\tdependency\tresource\n' > "$ppg_manifest"
printf 'smoke\t%s\tnone\trtx_5080\n' "$ppg_smoke" >> "$ppg_manifest"
mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")"
trap 'cp "$ppg_manifest" "$ppg_nas/outputs/$(basename "$ppg_root")/"' EXIT
submit() {
  local stage="$1" dependency="$2" gpu="$3" duration="$4"
  local options=(--parsable --export=ALL --job-name="ppg_enroll_${stage}" --kill-on-invalid-dep=yes --dependency="afterok:$dependency" --time="$duration")
  [[ "$gpu" == none ]] || options+=(--gres="gpu:$gpu:1")
  local id
  id="$(sbatch "${options[@]}" "$ppg_code/scripts/sbatch_post_enrollment.sh" "$stage" "$ppg_root" "$ppg_store")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\n' "$stage" "$id" "$dependency" "$gpu" >> "$ppg_manifest"
  printf '%s' "$id"
}
prepare="$(submit prepare "$ppg_smoke" none 02:00:00)"
inner="$(submit population_inner "$prepare" rtx_5080 72:00:00)"
final="$(submit population_final "$inner" rtx_5080 72:00:00)"
benchmark="$(submit population_benchmark "$final" rtx_5080 03:00:00)"
p0="$(submit personal_0 "$benchmark" rtx_5080 12:00:00)"
p1="$(submit personal_1 "$benchmark" rtx_5070_ti 12:00:00)"
evaluate="$(submit evaluate "$p0:$p1" none 02:00:00)"
printf 'FINAL_EVALUATOR=%s\n' "$evaluate"
cat "$ppg_manifest"
