#!/bin/bash
#SBATCH --job-name=ppg_torch_install
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"

python -m pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cu130
python - <<'PY'
import torch
print(f"PYTORCH_VERSION={torch.__version__}")
print(f"COMPILED_CUDA_RUNTIME={torch.version.cuda}")
assert torch.__version__.startswith("2.13.0")
PY
conda env export -p "$work_root/envs/train" > "$archive_root/environment.yml"
echo "PYTORCH_INSTALL_COMPLETE=yes"
