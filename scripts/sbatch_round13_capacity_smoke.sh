#!/bin/bash
#SBATCH --job-name=ppg_r13_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:-$work_root/code/pulsedb_fewshot}"
run_id="event120-v1_round13_capacity_smoke_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$project_root"
mkdir -p "$output"

python - "$output/smoke.json" <<'PY'
import gc
import json
from pathlib import Path
import sys

import torch

from pulsedb_fewshot.models import (
    PopulationRegressor,
    VariableKPersonalizer,
    build_ppg_encoder,
    model_parameter_counts,
)

CANDIDATES = (
    "resnet_small",
    "resnet_depth2",
    "resnet_wide1p5",
    "inception_time",
    "inception_time_wide",
    "patch_transformer",
    "patch_transformer_deep",
    "patch_transformer_wide",
    "patch_transformer_highres",
    "patch_transformer_longpatch",
    "conformer",
    "conformer_large",
    "convnext_1d",
)
POPULATION_MICROBATCHES = (32,) * len(CANDIDATES)
QGH_MICROBATCHES = (16,) * len(CANDIDATES)
EXPECTED_POPULATION_PARAMETERS = {
    "resnet_small": 665_490,
    "resnet_depth2": 1_333_010,
    "resnet_wide1p5": 1_456_282,
    "inception_time": 512_162,
    "inception_time_wide": 1_123_954,
    "patch_transformer": 710_530,
    "patch_transformer_deep": 1_372_034,
    "patch_transformer_wide": 2_730_498,
    "patch_transformer_highres": 715_522,
    "patch_transformer_longpatch": 716_034,
    "conformer": 1_587_330,
    "conformer_large": 5_230_018,
    "convnext_1d": 5_477_826,
}

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated Slurm job")

device = torch.device("cuda")
records = []
for backbone, population_batch, qgh_batch in zip(
    CANDIDATES, POPULATION_MICROBATCHES, QGH_MICROBATCHES, strict=True
):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    encoder = build_ppg_encoder(backbone, feature_dim=256)
    population = PopulationRegressor(encoder).to(device).train()
    population_counts = model_parameter_counts(population)
    if population_counts["total"] != EXPECTED_POPULATION_PARAMETERS[backbone]:
        raise AssertionError(f"unexpected population parameter count for {backbone}")
    optimizer = torch.optim.AdamW(population.parameters(), lr=1e-4)
    inputs = torch.randn(population_batch, 1, 1250, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    ):
        population_output = population(inputs)
        population_loss = population_output.square().mean()
    population_loss.backward()
    optimizer.step()
    population_finite = bool(torch.isfinite(population_output).all())
    population_peak = round(torch.cuda.max_memory_allocated() / 2**20, 2)

    del inputs, optimizer, population_loss, population_output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for parameter in population.parameters():
        parameter.requires_grad = False
    personalizer = VariableKPersonalizer(
        population,
        use_film=False,
        query_conditioned_weights=False,
        use_quality_gate=True,
    ).to(device).train()
    personalizer.population.eval()
    personal_counts = model_parameter_counts(personalizer)
    if personal_counts["trainable"] != 461_828:
        raise AssertionError(f"unexpected QGH trainable count for {backbone}")
    if personal_counts["total"] != population_counts["total"] + 461_828:
        raise AssertionError(f"unexpected QGH total count for {backbone}")
    personal_parameters = [
        parameter for parameter in personalizer.parameters() if parameter.requires_grad
    ]
    personal_optimizer = torch.optim.AdamW(personal_parameters, lr=3e-4)
    support_ppg = torch.randn(qgh_batch, 5, 1, 1250, device=device)
    support_bp = torch.randn(qgh_batch, 5, 2, device=device)
    support_mask = torch.ones(qgh_batch, 5, dtype=torch.bool, device=device)
    query_ppg = torch.randn(qgh_batch, 1, 1250, device=device)
    personal_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    ):
        personal_output = personalizer(
            query_ppg,
            support_ppg,
            support_bp,
            support_mask,
        )
        personal_loss = personal_output.square().mean()
    personal_loss.backward()
    personal_optimizer.step()
    qgh_peak = round(torch.cuda.max_memory_allocated() / 2**20, 2)

    record = {
        "backbone": backbone,
        "population_microbatch": population_batch,
        "qgh_microbatch": qgh_batch,
        "population_output_shape": [population_batch, 2],
        "personal_output_shape": list(personal_output.shape),
        "finite": bool(
            population_finite and torch.isfinite(personal_output).all()
        ),
        "population_parameters": population_counts,
        "qgh_parameters": personal_counts,
        "population_peak_gpu_memory_mib": population_peak,
        "qgh_peak_gpu_memory_mib": qgh_peak,
    }
    if not record["finite"]:
        raise RuntimeError(f"non-finite output for {backbone}")
    records.append(record)

    del (
        personal_loss,
        personal_output,
        personal_optimizer,
        personal_parameters,
        personalizer,
        population,
        encoder,
        query_ppg,
        support_bp,
        support_mask,
        support_ppg,
    )

payload = {
    "status": "pass",
    "cuda_device": torch.cuda.get_device_name(device),
    "candidate_count": len(records),
    "records": records,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
print("ROUND13_CAPACITY_GPU_SMOKE=pass")
PY

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
echo "ROUND13_CAPACITY_SMOKE_COMPLETE=$run_id"
