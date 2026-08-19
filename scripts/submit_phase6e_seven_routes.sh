#!/bin/bash
set -euo pipefail
work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; cd $root
pipeline=$work/outputs/event120-v1_phase6e_continuous
if [[ -e $pipeline ]]; then
  [[ ! -e $pipeline/waveform_embeddings.parquet ]] || { echo "ERROR: completed Phase-6E embedding artifact already exists" >&2; exit 1; }
  echo "RESUME_FROM_FAILED_PREPARE=yes"
else
  mkdir $pipeline
fi
seed=20260820; prepare=$(sbatch --parsable scripts/sbatch_phase6e_prepare.sh)
declare -a jobs specs
ridge=$(sbatch --parsable --dependency=afterok:$prepare --job-name=ppg_p6e_ridge scripts/sbatch_phase6e_train.sh ridge $seed); jobs+=($ridge); specs+=("R6-1 Ridge residual=$work/outputs/event120-v1_phase6e_ridge_seed${seed}_job${ridge}")
methods=(mlp gated_mlp weighted_mlp causal_gru supervised_moe); labels=("R6-2 Residual MLP" "R6-3 Gated residual MLP" "R6-4 Difficult-2x residual MLP" "R6-5 Causal GRU residual" "R6-6 Supervised MoE")
for i in ${!methods[@]}; do if ((i%2==0)); then gpu=rtx_5070_ti; else gpu=rtx_5080; fi; job=$(sbatch --parsable --dependency=afterok:$prepare --gres=gpu:$gpu:1 --job-name=ppg_p6e_${methods[$i]} scripts/sbatch_phase6e_train.sh ${methods[$i]} $seed); jobs+=($job); specs+=("${labels[$i]}=$work/outputs/event120-v1_phase6e_${methods[$i]}_seed${seed}_job${job}"); done
embed=$(sbatch --parsable --dependency=afterok:$prepare scripts/sbatch_phase6e_embeddings.sh); cluster=$(sbatch --parsable --dependency=afterok:$embed scripts/sbatch_phase6e_cluster.sh $seed); jobs+=($cluster); specs+=("R6-7 Morphology cluster MoE=$work/outputs/event120-v1_phase6e_morphology_cluster_moe_seed${seed}_job${cluster}")
dependency=$(IFS=:; echo "${jobs[*]}"); report=$(sbatch --parsable --dependency=afterok:$dependency scripts/sbatch_phase6e_report.sh "${specs[@]}")
manifest=$work/outputs/submission_manifests/event120-v1_phase6e_seven_routes_$(date +%Y%m%d-%H%M%S).txt
{ echo PHASE=Phase-6E_seven_route_continuous_screen; echo LOCKED_META_TEST_ACCESSED=no; echo PREPARE_JOB=$prepare; for i in ${!jobs[@]}; do echo CANDIDATE_$i=${jobs[$i]}; done; echo EMBEDDING_JOB=$embed; echo CLUSTER_JOB=$cluster; echo REPORT_JOB=$report; echo PROMOTION_GATE=overall_gain_at_least_0.15_mmHg_and_both_sources_improve; } | tee $manifest
mkdir -p $archive/outputs/submission_manifests; cp $manifest $archive/outputs/submission_manifests/
