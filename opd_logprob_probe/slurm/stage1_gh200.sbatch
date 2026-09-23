#!/bin/bash
#SBATCH --job-name=opd-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Add cluster-specific --account and --partition lines if required.
set -euo pipefail
nvidia-smi || true
# Activate your cluster's known-good GH200 PyTorch env/container here.
python scripts/check_gpu_env.py
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python stage1_collect.py --config configs/rollout_aime.yaml
