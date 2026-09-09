#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=72:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/post_enrollment_%j.log
set -euo pipefail
umask 077
ppg_code="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_code"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_python="$ppg_work/envs/train/bin/python"
ppg_stage="${1:?stage}"; ppg_root="${2:?batch root}"; ppg_store="${3:?source store}"
[[ "$ppg_root" == "$ppg_work/outputs/post-enrollment-30-v1_"* ]]
[[ "$ppg_store" == "$ppg_work/data/processed/pulsedb-official-calbased-v1_"* ]]
export PYTHONPATH="$ppg_code/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$ppg_root"
ppg_output="$ppg_root/$ppg_stage"
ppg_plan="$ppg_root/prepare/plan.json"
archive() {
  local code=$?
  trap - EXIT
  mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")" "$ppg_nas/logs"
  if [[ -d "$ppg_output" ]]; then
    mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")/$(basename "$ppg_output")"
    rsync -a --no-perms --no-owner --no-group "$ppg_output/" "$ppg_nas/outputs/$(basename "$ppg_root")/$(basename "$ppg_output")/" || code=1
  fi
  cp "$ppg_work/logs/post_enrollment_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap archive EXIT
cd "$ppg_code"
case "$ppg_stage" in
  smoke)
    mkdir "$ppg_output"
    "$ppg_python" -m unittest discover -s tests -p 'test_post_enrollment.py' -v
    "$ppg_python" -m unittest discover -s tests -p 'test_official_calbased_train.py' -v
    "$ppg_python" -c 'import torch; from pulsedb_fewshot.lora_prs_models import LoraPRSRegressor, PRS_MODELS; from pulsedb_fewshot.post_enrollment_personal import fresh_person_model, feature_forward; assert torch.cuda.is_available(); base=LoraPRSRegressor(PRS_MODELS["lora_continue"],subject_count=2).cuda(); m=fresh_person_model(base.state_dict(),20260909,"cuda"); x=torch.randn(4,1,1250,device="cuda"); z=m.base.encoder(x).detach(); y,_=feature_forward(m,z,torch.zeros(4,2,device="cuda")); y.square().mean().backward(); assert m.base.lora_b.weight.grad is not None; assert all(p.grad is None for p in m.base.encoder.parameters()); print("GPU_PERSONAL_FORWARD_BACKWARD=pass",torch.cuda.get_device_name(0))'
    ;;
  prepare)
    "$ppg_python" -m pulsedb_fewshot.post_enrollment_protocol --store-root "$ppg_store" --output "$ppg_output"
    chmod 400 "$ppg_output/plan.json" "$ppg_output/"*.parquet
    ;;
  population_inner)
    "$ppg_python" -m pulsedb_fewshot.post_enrollment_population --plan "$ppg_plan" --output "$ppg_output" --stage inner
    ;;
  population_final)
    "$ppg_python" -m pulsedb_fewshot.post_enrollment_population --plan "$ppg_plan" --output "$ppg_output" --stage final --selection-run "$ppg_root/population_inner"
    ;;
  population_benchmark)
    "$ppg_python" -m pulsedb_fewshot.post_enrollment_evaluate --stage population --plan "$ppg_plan" --population-run "$ppg_root/population_final" --full-index "$ppg_work/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet" --output "$ppg_output"
    ;;
  personal_0|personal_1)
    "$ppg_python" -m pulsedb_fewshot.post_enrollment_personal --plan "$ppg_plan" --population-run "$ppg_root/population_final" --output "$ppg_output" --shard "${ppg_stage#personal_}"
    ;;
  evaluate)
    "$ppg_python" -m pulsedb_fewshot.post_enrollment_evaluate --stage enrollment --plan "$ppg_plan" --population-run "$ppg_root/population_final" --personal-runs "$ppg_root/personal_0" "$ppg_root/personal_1" --full-index "$ppg_work/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet" --output "$ppg_output" --device cpu
    ;;
  *) printf 'Unknown enrollment stage: %s\n' "$ppg_stage" >&2; exit 2 ;;
esac
