#!/bin/bash
#SBATCH --job-name=ppg_torch_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_5080:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

python -c 'import torch'

python - <<'PY'
import os
import torch
print(f"PYTORCH_VERSION={torch.__version__}")
print(f"CUDA_RUNTIME={torch.version.cuda}")
print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
print(f"GPU_NAME={torch.cuda.get_device_name(0)}")
assert torch.cuda.is_available()
PY

python -m pytest
python - <<'PY'
import torch
from pulsedb_fewshot.models import VariableKPersonalizer

device = torch.device("cuda")
model = VariableKPersonalizer(
    use_film=True, query_conditioned_weights=True
).to(device)
query = torch.randn(2, 1, 1250, device=device)
support = torch.randn(2, 5, 1, 1250, device=device)
support_bp = torch.randn(2, 5, 2, device=device)
mask = torch.tensor(
    [[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool, device=device
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
prediction = model(query, support, support_bp, mask)
loss = prediction.square().mean()
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
checkpoint = f"/home/{os.environ['USER']}/work/ppg_bp/checkpoints/gpu_smoke.pt"
torch.save({"model": model.state_dict(), "loss": float(loss.detach())}, checkpoint)
print(f"GPU_FORWARD_BACKWARD_CHECKPOINT=pass:{checkpoint}")
PY

conda env export -p "$work_root/envs/train" > "$archive_root/environment.yml"
echo "PYTORCH_GPU_SMOKE_COMPLETE=yes"
