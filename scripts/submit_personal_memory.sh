#!/bin/bash
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work="/home/jiangyu.cheng/work/ppg_bp"
ppg_nas="/home/jiangyu.cheng/nas/ppg_bp"
ppg_tag="$(date +%Y%m%d-%H%M%S)"
ppg_manifest="$ppg_work/outputs/submission_manifests/personal_memory_${ppg_tag}.tsv"
mkdir -p "$(dirname "$ppg_manifest")" "$ppg_nas/outputs/submission_manifests"
printf 'stage\tsplit_mode\tcandidate\tjob_id\toutput\n' > "$ppg_manifest"
trap 'cp "$ppg_manifest" "$ppg_nas/outputs/submission_manifests/"' EXIT
declare -A caches gpus
caches_jobs=()
gpus[random_disjoint]="gpu:rtx_5080:1"
gpus[chronological_blocked]="gpu:rtx_5070_ti:1"
ppg_submit() {
  local stage="$1" mode="$2" candidate="$3" dependency="$4"; shift 4
  local output="$ppg_work/outputs/personal-memory-v1_${ppg_tag}_${mode}_${candidate}"
  local options=(--parsable --export="ALL,PPG_PROJECT_ROOT=$ppg_project" --job-name="ppg_mem_${stage}_${mode:0:1}")
  if [[ "$dependency" != none ]]; then options+=(--dependency="afterok:$dependency"); fi
  if [[ "$stage" == smoke || "$stage" == cache || "$stage" == train ]]; then options+=(--gres="${gpus[$mode]}"); fi
  if [[ "$stage" == smoke ]]; then options+=(--time=00:20:00); fi
  if [[ "$stage" == gate || "$stage" == report || "$stage" == final ]]; then options+=(--time=01:00:00 --mem=8G); fi
  local id
  id="$(sbatch "${options[@]}" "$ppg_project/scripts/sbatch_personal_memory.sh" "$stage" "$mode" "$output" "$@")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$mode" "$candidate" "$id" "$output" >> "$ppg_manifest"
  printf '%s' "$id"
}
for mode in random_disjoint chronological_blocked; do
  smoke="$(ppg_submit smoke "$mode" smoke none)"
  cache="$(ppg_submit cache "$mode" cache "$smoke")"
  caches_jobs+=("$cache")
  caches[$mode]="$ppg_work/outputs/personal-memory-v1_${ppg_tag}_${mode}_cache"
done
gate="$(ppg_submit gate both gate "${caches_jobs[0]}:${caches_jobs[1]}" "${caches[random_disjoint]}" "${caches[chronological_blocked]}")"
gate_path="$ppg_work/outputs/personal-memory-v1_${ppg_tag}_both_gate/gate.json"
reports=(); report_jobs=()
for mode in random_disjoint chronological_blocked; do
  jobs=(); runs=()
  for candidate in lora_continued_control pair_single pair_uniform pair_retrieved pair_distance_blend; do
    id="$(ppg_submit train "$mode" "$candidate" "$gate" "$candidate" "${caches[$mode]}" "$gate_path")"
    jobs+=("$id")
    runs+=("$ppg_work/outputs/personal-memory-v1_${ppg_tag}_${mode}_${candidate}")
  done
  dependency="$(IFS=:; echo "${jobs[*]}")"
  id="$(ppg_submit report "$mode" report "$dependency" "${caches[$mode]}" "${runs[@]}")"
  report_jobs+=("$id")
  reports+=("$ppg_work/outputs/personal-memory-v1_${ppg_tag}_${mode}_report")
done
ppg_submit final both final "${report_jobs[0]}:${report_jobs[1]}" "${reports[@]}"
printf '\nMANIFEST=%s\n' "$ppg_manifest"
cat "$ppg_manifest"
