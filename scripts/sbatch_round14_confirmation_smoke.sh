#!/bin/bash
#SBATCH --job-name=ppg_r14_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name the immutable Round-14 snapshot}"
run_id="event120-v1_round14_confirmation_smoke_job${SLURM_JOB_ID}"
output="$work_root/outputs/$run_id"

test -d "$project_root"
test ! -w "$project_root"
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

CANDIDATES = ("resnet_small", "inception_time_wide")
EXPECTED_POPULATION_PARAMETERS = {
    "resnet_small": 665_490,
    "inception_time_wide": 1_123_954,
}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated Slurm job")

device = torch.device("cuda")
records = []
for backbone in CANDIDATES:
    gc.collect()
    torch.cuda.empty_cache()
    encoder = build_ppg_encoder(backbone, feature_dim=256)
    population = PopulationRegressor(encoder).to(device).train()
    population_counts = model_parameter_counts(population)
    if population_counts["total"] != EXPECTED_POPULATION_PARAMETERS[backbone]:
        raise AssertionError(f"unexpected population parameters for {backbone}")
    population_optimizer = torch.optim.AdamW(population.parameters(), lr=1e-4)
    population_input = torch.randn(32, 1, 1250, device=device)
    population_optimizer.zero_grad(set_to_none=True)
    population_output = population(population_input)
    population_output.square().mean().backward()
    population_optimizer.step()

    del population_input, population_optimizer, population_output
    gc.collect()
    torch.cuda.empty_cache()
    for parameter in population.parameters():
        parameter.requires_grad = False
    personalizer = VariableKPersonalizer(
        population,
        use_film=False,
        query_conditioned_weights=False,
        use_quality_gate=True,
    ).to(device).train()
    personalizer.population.eval()
    qgh_counts = model_parameter_counts(personalizer)
    if qgh_counts["trainable"] != 461_828:
        raise AssertionError(f"unexpected QGH trainable parameters for {backbone}")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in personalizer.parameters() if parameter.requires_grad],
        lr=3e-4,
    )
    support_ppg = torch.randn(16, 5, 1, 1250, device=device)
    support_bp = torch.randn(16, 5, 2, device=device)
    support_mask = torch.ones(16, 5, dtype=torch.bool, device=device)
    query_ppg = torch.randn(16, 1, 1250, device=device)
    optimizer.zero_grad(set_to_none=True)
    prediction = personalizer(query_ppg, support_ppg, support_bp, support_mask)
    prediction.square().mean().backward()
    optimizer.step()
    if prediction.shape != (16, 2) or not torch.isfinite(prediction).all():
        raise RuntimeError(f"invalid QGH output for {backbone}")
    records.append(
        {
            "backbone": backbone,
            "population_microbatch": 32,
            "qgh_microbatch": 16,
            "population_parameters": population_counts,
            "qgh_parameters": qgh_counts,
        }
    )
    del (
        encoder,
        optimizer,
        personalizer,
        population,
        prediction,
        query_ppg,
        support_bp,
        support_mask,
        support_ppg,
    )

payload = {
    "status": "pass",
    "cuda_device": torch.cuda.get_device_name(device),
    "records": records,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
print("ROUND14_CONFIRMATION_GPU_SMOKE=pass")
PY

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id" >/dev/null
echo "ROUND14_CONFIRMATION_SMOKE_COMPLETE=$run_id"
