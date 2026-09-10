#!/bin/bash
# Resource sizes are assigned explicitly by submit_full_enrollment.sh.
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/full_enrollment_%j.log
set -euo pipefail
umask 077
ppg_code="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_code"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_stage="${1:?stage}"
ppg_root="${2:?batch root}"
[[ "$ppg_root" == "$ppg_work/outputs/full-cohort-enrollment-v1_"* ]]
ppg_tag="${ppg_root##*full-cohort-enrollment-v1_}"
[[ "$ppg_tag" =~ ^[0-9]{8}-[0-9]{6}$ ]]
ppg_store="$ppg_work/data/processed/full-cohort-inputs-v1_$ppg_tag"
ppg_split="$ppg_work/data/manifests/pulsedb_v2_full_cohort/subject_splits.csv"
ppg_index="$ppg_work/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet"
ppg_raw="$ppg_work/data/raw/PulseDB_v2/Segment_Files"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ppg_work/envs/train"
ppg_python="$ppg_work/envs/train/bin/python"
export PYTHONPATH="$ppg_code/src" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$ppg_root"
ppg_output="$ppg_root/$ppg_stage"
ppg_plan="$ppg_root/prepare/plan.json"
archive() {
  local status=$?
  trap - EXIT
  mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")" "$ppg_nas/logs"
  if [[ -d "$ppg_output" ]]; then
    mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")/$ppg_stage"
    rsync -a --no-perms --no-owner --no-group "$ppg_output/" "$ppg_nas/outputs/$(basename "$ppg_root")/$ppg_stage/" || status=1
  fi
  cp "$ppg_work/logs/full_enrollment_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$status"
}
trap archive EXIT
cd "$ppg_code"
case "$ppg_stage" in
  smoke)
    mkdir "$ppg_output"
    for ppg_test in test_full_enrollment.py test_legacy_enrollment.py test_post_enrollment.py test_post_enrollment_ablations.py test_personal_memory_prepare.py; do
      "$ppg_python" -m unittest discover -s tests -p "$ppg_test" -v
    done
    "$ppg_python" -c 'import torch; from pulsedb_fewshot.lora_prs_models import LoraPRSRegressor,PRS_MODELS; from pulsedb_fewshot.post_enrollment_personal import fresh_person_model,feature_forward; assert torch.cuda.is_available(); m=LoraPRSRegressor(PRS_MODELS["lora_continue"],subject_count=2).cuda(); x=torch.randn(4,1,1250,device="cuda"); y=m(x,torch.zeros(4,2,device="cuda"),subject_index=torch.zeros(4,dtype=torch.long,device="cuda")); (y-1).square().mean().backward(); p=fresh_person_model(m.state_dict(),20260910,"cuda"); z=p.base.encoder(x).detach(); out,_=feature_forward(p,z,torch.zeros(4,2,device="cuda")); out.square().mean().backward(); assert p.base.lora_b.weight.grad is not None; assert all(v.grad is None for v in p.base.encoder.parameters()); print("GPU_FORWARD_BACKWARD=pass",torch.cuda.get_device_name(0))'
    "$ppg_python" -c 'import os,sys; from pathlib import Path; from pulsedb_fewshot.official_calbased_train import save_json; from pulsedb_fewshot.training import source_tree_sha256; save_json(Path(sys.argv[1]),{"status":"pass","protocol_id":"full-cohort-enrollment-v1","snapshot":os.environ["PPG_PROJECT_ROOT"],"source_tree_sha256":source_tree_sha256(Path(os.environ["PPG_PROJECT_ROOT"])),"slurm_job_id":os.environ["SLURM_JOB_ID"],"gpu_test_passed":True})' "$ppg_output/receipt.json"
    ;;
  materialize)
    mkdir "$ppg_output"
    "$ppg_python" -m pulsedb_fewshot.full_enrollment_data --stage materialize --full-index "$ppg_index" --legacy-split "$ppg_split" --raw-root "$ppg_raw" --output "$ppg_store" --workers 2 --shards 32
    cp "$ppg_store/manifest.json" "$ppg_output/"
    chmod 400 "$ppg_store/"*.json "$ppg_store/"*.parquet "$ppg_store/"*.npy
    ;;
  prepare)
    "$ppg_python" -m pulsedb_fewshot.full_enrollment_data --stage prepare --store-root "$ppg_store" --legacy-split "$ppg_split" --full-index "$ppg_index" --output "$ppg_output"
    chmod 400 "$ppg_output/"*.json "$ppg_output/"*.parquet "$ppg_output/"*.csv
    ;;
  population)
    "$ppg_python" -m pulsedb_fewshot.legacy_enrollment_train --stage population --plan "$ppg_plan" --output "$ppg_output"
    ;;
  validation_0|validation_1|test_0|test_1)
    "$ppg_python" -m pulsedb_fewshot.legacy_enrollment_train --stage personal --cohort "${ppg_stage%_*}" --shard "${ppg_stage##*_}" --plan "$ppg_plan" --population-run "$ppg_root/population" --validation-run "$ppg_root/validation_score" --output "$ppg_output" --workers 0
    ;;
  validation_score|test_score)
    ppg_cohort="${ppg_stage%_score}"
    "$ppg_python" -m pulsedb_fewshot.legacy_enrollment_evaluate --cohort "$ppg_cohort" --plan "$ppg_plan" --population-run "$ppg_root/population" --personal-runs "$ppg_root/${ppg_cohort}_0" "$ppg_root/${ppg_cohort}_1" --output "$ppg_output"
    ;;
  *) printf 'Unknown stage: %s\n' "$ppg_stage" >&2; exit 2 ;;
esac
