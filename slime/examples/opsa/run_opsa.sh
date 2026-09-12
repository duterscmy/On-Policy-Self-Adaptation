#!/bin/bash
#SBATCH --job-name="slime_dynamic"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/u6os/public/mingyu/opsa/slime-legacy
#SBATCH -o slurm_dynamic.%j.%N.out
#SBATCH -e slurm_dynamic.%j.%N.err

set -eo pipefail

# ------------------------------------------------------------
# Conda
# ------------------------------------------------------------
source /home/u6os/cmy9797.u6os/miniconda3/etc/profile.d/conda.sh
conda activate opsa

# ------------------------------------------------------------
# GH200 / aarch64 CUDA 12.6 runtime
# ------------------------------------------------------------
export HPC_SDK=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11
export CUDA_HOME="$HPC_SDK/cuda/12.6"
export MATHLIB_HOME="$HPC_SDK/math_libs/12.6"
export NCCL_INCLUDE="$HPC_SDK/comm_libs/12.6/nccl/include"
export PATH="$CUDA_HOME/bin:/usr/sbin:/sbin:$PATH"

export CUDNN_ROOT="$(python - <<'PY_CUDNN'
from importlib.metadata import distribution
print(distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn"))
PY_CUDNN
)"

# Include the sbsa target headers/libs as well; TE 2.4 needs cublas/cusparse here.
export CUDA_TARGET_INCLUDE="$CUDA_HOME/targets/sbsa-linux/include"
export CUDA_TARGET_LIB="$CUDA_HOME/targets/sbsa-linux/lib"

export CPATH="$NCCL_INCLUDE:$CUDNN_ROOT/include:$MATHLIB_HOME/include:$CUDA_TARGET_INCLUDE:$CUDA_HOME/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CPATH"
export LIBRARY_PATH="$CUDNN_ROOT/lib:$MATHLIB_HOME/lib64:$CUDA_TARGET_LIB:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDNN_ROOT/lib:$MATHLIB_HOME/lib64:$CUDA_TARGET_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# ------------------------------------------------------------
# Legacy Slime / Megatron
# ------------------------------------------------------------
export SLIME_ROOT=/projects/u6os/public/mingyu/opsa/slime-legacy
export MEGATRON_PATH=/projects/u6os/public/mingyu/models/Megatron-LM
export PYTHONPATH="$SLIME_ROOT:$MEGATRON_PATH:${PYTHONPATH:-}"

cd "$SLIME_ROOT"

echo "============================================================"
echo "Slurm job:        ${SLURM_JOB_ID:-unknown}"
echo "Host:             $(hostname)"
echo "Slime:            $SLIME_ROOT"
echo "Megatron:         $MEGATRON_PATH"
echo "CUDA_VISIBLE:     ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

nvidia-smi -L

# ------------------------------------------------------------
# Preflight: fail before training if TE/GPU environment is broken
# ------------------------------------------------------------
python - <<'PY_PREFLIGHT'
import torch
import transformer_engine
import transformer_engine.pytorch

print("[preflight] torch:", torch.__version__)
print("[preflight] CUDA:", torch.version.cuda)
print("[preflight] visible GPUs:", torch.cuda.device_count())
print("[preflight] TE:", transformer_engine.__version__)

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= 4, f"Expected 4 GPUs, got {torch.cuda.device_count()}"
print("[preflight] Transformer Engine PyTorch backend: OK")
PY_PREFLIGHT

# ------------------------------------------------------------
# Locate the legacy experiment script
# ------------------------------------------------------------
TRAIN_SCRIPT="$(find "$SLIME_ROOT" -type f \
  -name 'run_train_gsm8k.rollout8.block1.dynamic_sampling_control.sh' \
  -print -quit)"

if [ -z "$TRAIN_SCRIPT" ]; then
    echo "ERROR: could not find run_train_gsm8k.rollout8.block1.dynamic_sampling_control.sh"
    exit 2
fi

echo "Training script: $TRAIN_SCRIPT"

# ------------------------------------------------------------
# Force Transformer Engine.
#
# If the script explicitly says '--transformer-impl local', create a patched
# sibling copy without touching the original.  If it forwards "$@", pass the
# option normally.  Otherwise run it unchanged and print a warning.
# ------------------------------------------------------------
RUN_SCRIPT="$TRAIN_SCRIPT"
EXTRA_ARGS=()

if grep -Eq -- '--transformer-impl[[:space:]]+local' "$TRAIN_SCRIPT"; then
    RUN_SCRIPT="${TRAIN_SCRIPT%.sh}.te24.slurm.sh"
    sed -E 's/--transformer-impl[[:space:]]+local/--transformer-impl transformer_engine/g' \
        "$TRAIN_SCRIPT" > "$RUN_SCRIPT"
    chmod +x "$RUN_SCRIPT"
    echo "Patched explicit '--transformer-impl local' -> transformer_engine"
elif grep -Eq '\$@|\$\{[@*]\}' "$TRAIN_SCRIPT"; then
    EXTRA_ARGS+=(--transformer-impl transformer_engine)
    echo "Passing '--transformer-impl transformer_engine' through script arguments"
else
    echo "WARNING: training script does not visibly forward \$@ and does not explicitly set"
    echo "         --transformer-impl local. Running with its own/default Megatron setting."
    echo "         If torch_norm persist_layer_norm appears again, inspect this script next."
fi

echo
echo "Starting legacy dynamic-sampling training..."
echo "Command: bash $RUN_SCRIPT ${EXTRA_ARGS[*]}"
echo

exec bash "$RUN_SCRIPT" "${EXTRA_ARGS[@]}"
