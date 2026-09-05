#!/bin/bash
#SBATCH --job-name=ppg_feature_smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "/home/$USER/work/ppg_bp/envs/train"
ppg_project="${PPG_PROJECT_ROOT:?snapshot required}"
test ! -w "$ppg_project"
export PYTHONPATH="$ppg_project/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PPG_TEST_DEVICE=cuda
cd "$ppg_project"
python -c 'import torch; print(torch.cuda.get_device_name(0))'
python -m pytest tests/test_personal_feature_models.py -q -p no:cacheprovider
echo "PERSONAL_FEATURE_CUDA_SMOKE=pass"
