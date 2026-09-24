#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Runtime environment setup for a fresh Arrhenius GPU node
# Do NOT load GPU/buildenv-* here; those modules were only for compilation.
# ============================================================
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate opsa-slime

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export OPSA_ROOT="${OPSA_ROOT:-/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa}"
export SLIME_DIR="${SLIME_DIR:-$OPSA_ROOT/slime-upstream}"
export SGLANG_DIR="${SGLANG_DIR:-$OPSA_ROOT/sglang}"
export MEGATRON_DIR="${MEGATRON_DIR:-$OPSA_ROOT/Megatron-LM}"
export HF_CKPT="${HF_CKPT:-$OPSA_ROOT/models/Qwen3-1.7B}"
export MCORE_CKPT="${MCORE_CKPT:-$OPSA_ROOT/models/Qwen3-1.7B_torch_dist}"
export DATA_ROOT="${DATA_ROOT:-$OPSA_ROOT/data/dapo-math-17k}"

# CUDA toolkit
export CUDA_HOME="${CUDA_HOME:-/software/sse2/el9_gh200/easybuild/prefix/software/CUDA/12.9.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# Runtime libraries needed by TE / Apex / PyTorch on a fresh node.
# cuDNN 9.17.1.4 is installed inside this conda env.
export CUDNN_ROOT="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn"
export NCCL_HOME="/software/sse2/el9_gh200/easybuild/prefix/software/NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1"
export GCC14_LIB="/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64"
export TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$GCC14_LIB:$CUDNN_ROOT/lib:$NCCL_HOME/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# Make sure Ray workers resolve the patched source trees, not stale packages.
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SGLANG_DIR/python:${PYTHONPATH:-}"

echo "========== Runtime environment =========="
echo "host:        $(hostname)"
echo "conda:       $CONDA_PREFIX"
echo "python:      $(which python)"
echo "CUDA_HOME:   $CUDA_HOME"
echo "nvcc:        $(which nvcc)"
echo "PYTHONPATH:  $PYTHONPATH"
echo "GPUs:"
nvidia-smi -L
echo "========================================="

# Fast preflight: fail before Ray startup if the environment is broken.
python - <<'PY_PREFLIGHT'
import torch
import transformer_engine
import apex
import sglang
import megatron.core
import slime
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count())
print("SGLang:", sglang.__version__)
print("TE:", transformer_engine.__version__)
print("preflight: OK")
PY_PREFLIGHT

ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES="${N_SAMPLES:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"

if [[ -f "$DATA_ROOT/smoke-8.jsonl" ]]; then
  PROMPT_DATA="${PROMPT_DATA:-$DATA_ROOT/smoke-8.jsonl}"
else
  PROMPT_DATA="${PROMPT_DATA:-$DATA_ROOT/dapo-math-17k.jsonl}"
fi

for path in "$SLIME_DIR" "$MEGATRON_DIR" "$HF_CKPT" "$MCORE_CKPT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
[[ -f "$PROMPT_DATA" ]] || { echo "ERROR: missing prompt data $PROMPT_DATA" >&2; exit 1; }

VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
NEEDED_GPUS="$((ACTOR_GPUS + ROLLOUT_GPUS))"
(( VISIBLE_GPUS >= NEEDED_GPUS )) || { echo "ERROR: visible GPUs=$VISIBLE_GPUS, need $NEEDED_GPUS" >&2; exit 1; }
(( ACTOR_GPUS % TP_SIZE == 0 )) || { echo "ERROR: ACTOR_GPUS must be divisible by TP_SIZE" >&2; exit 1; }

mkdir -p "$OPSA_ROOT/logs"
LOG_FILE="$OPSA_ROOT/logs/qwen3-1.7b-opd-smoke-$(date +%Y%m%d-%H%M%S).log"

echo "========== OPD smoke =========="
echo "node: $(hostname)"
echo "visible GPUs: $VISIBLE_GPUS"
echo "actor GPUs: $ACTOR_GPUS"
echo "rollout GPUs: $ROLLOUT_GPUS"
echo "TP size: $TP_SIZE"
echo "HF: $HF_CKPT"
echo "MCore: $MCORE_CKPT"
echo "data: $PROMPT_DATA"
echo "log: $LOG_FILE"
echo "==============================="

cd "$SLIME_DIR"
source scripts/models/qwen3-1.7B.sh

CKPT_ARGS=(
  --hf-checkpoint "$HF_CKPT"
  --ref-load "$MCORE_CKPT"
)

ROLLOUT_ARGS=(
  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --apply-chat-template
  --rollout-shuffle
  --num-rollout 1
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$MAX_RESPONSE_LEN"
  --rollout-temperature 1.0
  --num-steps-per-rollout 1
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)

RM_ARGS=( --rm-type math )

PERF_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

OPD_ARGS=(
  --advantage-estimator grpo
  --use-opd
  --opd-type megatron
  --opd-kl-coef 1.0
  --opd-teacher-load "$MCORE_CKPT"
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static 0.4
  --sglang-attention-backend triton
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

cleanup() {
  set +e
  ray stop --force >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

ray stop --force >/dev/null 2>&1 || true
sleep 2

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

ray start   --head   --node-ip-address="$MASTER_ADDR"   --num-gpus="$VISIBLE_GPUS"   --disable-usage-stats   --dashboard-host=127.0.0.1   --dashboard-port=8265

RUNTIME_ENV_JSON="$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "$PYTHONPATH",
    "CUDA_HOME": "$CUDA_HOME",
    "CUDA_PATH": "$CUDA_PATH",
    "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TOKENIZERS_PARALLELISM": "false"
  }
}
EOF
)"

set -x
ray job submit   --address="http://127.0.0.1:8265"   --runtime-env-json="$RUNTIME_ENV_JSON"   -- python3 train.py   --actor-num-nodes 1   --actor-num-gpus-per-node "$ACTOR_GPUS"   --rollout-num-gpus "$ROLLOUT_GPUS"   "${MODEL_ARGS[@]}"   "${CKPT_ARGS[@]}"   "${ROLLOUT_ARGS[@]}"   "${OPTIMIZER_ARGS[@]}"   "${OPD_ARGS[@]}"   "${PERF_ARGS[@]}"   "${SGLANG_ARGS[@]}"   "${MISC_ARGS[@]}"   "${RM_ARGS[@]}"   2>&1 | tee "$LOG_FILE"
set +x

echo "SMOKE TEST FINISHED"
echo "Log: $LOG_FILE"
