#!/bin/bash
#SBATCH --job-name=opd-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-analysis-%j.out
#SBATCH --error=slurm-analysis-%j.err
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python stage2_analyze.py --config configs/analysis.yaml
