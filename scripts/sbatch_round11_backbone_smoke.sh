#!/bin/bash
#SBATCH --job-name=ppg_r11_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python - <<'PY'
import json

import torch

from pulsedb_fewshot.models import BACKBONE_NAMES, build_ppg_encoder, model_parameter_counts

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated Slurm job")
device = torch.device("cuda")
records = []
for backbone in BACKBONE_NAMES:
    torch.cuda.empty_cache()
    model = build_ppg_encoder(backbone, feature_dim=256).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randn(2, 1, 1250, device=device)
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = output.square().mean()
    loss.backward()
    optimizer.step()
    records.append(
        {
            "backbone": backbone,
            "output_shape": list(output.shape),
            "finite": bool(torch.isfinite(output).all()),
            "loss": float(loss.detach().cpu()),
            "parameters": model_parameter_counts(model),
            "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        }
    )
    del inputs, output, loss, optimizer, model
    torch.cuda.reset_peak_memory_stats()
print(json.dumps(records, indent=2))
print("ROUND11_BACKBONE_GPU_SMOKE=pass")
PY
