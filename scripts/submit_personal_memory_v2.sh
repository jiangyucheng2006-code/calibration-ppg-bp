#!/bin/bash
# One finite batch; does not retry failed runs or alter existing submissions.
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work="/home/jiangyu.cheng/work/ppg_bp"
ppg_nas="/home/jiangyu.cheng/nas/ppg_bp"
ppg_tag="${PPG_BATCH_TAG:-$(date +%Y%m%d-%H%M%S)}"
[[ "$ppg_tag" =~ ^[0-9]{8}-[0-9]{6}$ ]]
ppg_manifest="$ppg_work/outputs/submission_manifests/personal_memory_v2_${ppg_tag}.tsv"
[[ ! -e "$ppg_manifest" ]]
[[ "$(hostname)" == slurm ]]
ppg_node_gres="$(sinfo -N -h -n hpc-2 -o %G)"
[[ "$ppg_node_gres" == *gpu:rtx_5080:1* && "$ppg_node_gres" == *gpu:rtx_5070_ti:1* ]]
ppg_v1="$ppg_work/outputs/personal-memory-v1_20260907-084550"
for mode in random_disjoint chronological_blocked; do
  test -f "${ppg_v1}_${mode}_cache/manifest.json"
  test -f "${ppg_v1}_${mode}_pair_distance_blend/best.pt"
  test -f "${ppg_v1}_${mode}_lora_continued_control/run.json"
done
test -f "${ppg_v1}_both_gate/gate.json"
mkdir -p "$(dirname "$ppg_manifest")" "$ppg_nas/outputs/submission_manifests"
printf 'stage\tsplit_mode\tjob_id\toutput\tdependency\n' > "$ppg_manifest"
trap 'cp "$ppg_manifest" "$ppg_nas/outputs/submission_manifests/"' EXIT
declare -A gpus diagnostics diag_jobs
gpus[random_disjoint]='gpu:rtx_5080:1'
gpus[chronological_blocked]='gpu:rtx_5070_ti:1'
submit() {
  local stage="$1" mode="$2" dependency="$3"; shift 3
  local output="$ppg_work/outputs/personal-memory-v2_${ppg_tag}_${mode}_${stage}"
  local options=(--parsable --export="ALL,PPG_PROJECT_ROOT=$ppg_project" --job-name="ppg_mem2_${stage}_${mode:0:1}")
  [[ ! -e "$output" ]]
  if [[ "$dependency" != none ]]; then options+=(--dependency="afterok:$dependency"); fi
  if [[ "$stage" == smoke || "$stage" == diagnostics || "$stage" == e1 || "$stage" == e2 ]]; then
    options+=(--gres="${gpus[$mode]}")
  fi
  if [[ "$stage" == smoke ]]; then options+=(--time=00:30:00); fi
  if [[ "$stage" == diagnostics ]]; then options+=(--time=04:00:00); fi
  if [[ "$stage" == report || "$stage" == final ]]; then options+=(--time=01:00:00 --mem=8G); fi
  local id
  id="$(sbatch "${options[@]}" "$ppg_project/scripts/sbatch_personal_memory_v2.sh" "$stage" "$mode" "$output" "$@")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$mode" "$id" "$output" "$dependency" >> "$ppg_manifest"
  printf '%s' "$id"
}
for mode in random_disjoint chronological_blocked; do
  smoke="$(submit smoke "$mode" none)"
  diag_jobs[$mode]="$(submit diagnostics "$mode" "$smoke")"
  diagnostics[$mode]="$ppg_work/outputs/personal-memory-v2_${ppg_tag}_${mode}_diagnostics"
done
reports=(); report_jobs=()
for mode in random_disjoint chronological_blocked; do
  # Both audited diagnostic jobs must finish before either neural experiment.
  dependency="${diag_jobs[random_disjoint]}:${diag_jobs[chronological_blocked]}"
  e1="$(submit e1 "$mode" "$dependency")"
  e2="$(submit e2 "$mode" "$dependency")"
  root="$ppg_work/outputs/personal-memory-v2_${ppg_tag}_${mode}"
  report_jobs+=("$(submit report "$mode" "$e1:$e2" "${diagnostics[$mode]}" "${root}_e1" "${root}_e2")")
  reports+=("${root}_report")
done
submit final both "${report_jobs[0]}:${report_jobs[1]}" "${reports[0]}" "${reports[1]}"
printf '\nMANIFEST=%s\n' "$ppg_manifest"
cat "$ppg_manifest"
