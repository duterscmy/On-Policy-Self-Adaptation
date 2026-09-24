#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen3-1.7B baseline OPSA end-to-end smoke test
#
# Default:
#   1 GPU  -> Megatron actor
#   1 GPU  -> SGLang rollout
#   OPSA   -> fixed negative advantage on bottom-20% actor-logp tokens
#
# To use 4 GPUs instead:
#   ACTOR_GPUS=2 ROLLOUT_GPUS=2 TP_SIZE=2 \
#     bash examples/opsa/run-qwen3-1.7B-opsa-smoke.sh
#
# To test entropy OPSA after fixed mode passes:
#   OPSA_MODE=entropy bash examples/opsa/run-qwen3-1.7B-opsa-smoke.sh
# ============================================================

# ---------- Runtime environment on a fresh Arrhenius GPU node ----------
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

# Runtime libraries needed by TE / Apex / PyTorch.
export CUDNN_ROOT="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn"
export NCCL_HOME="/software/sse2/el9_gh200/easybuild/prefix/software/NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1"
export GCC14_LIB="/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64"
export TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$GCC14_LIB:$CUDNN_ROOT/lib:$NCCL_HOME/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# Make Ray workers resolve the patched source trees.
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SGLANG_DIR/python:${PYTHONPATH:-}"

# ---------- Smoke-test configuration ----------
ACTOR_GPUS="${ACTOR_GPUS:-1}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES="${N_SAMPLES:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"

OPSA_MODE="${OPSA_MODE:-fixed}"
OPSA_TOKEN_FRACTION="${OPSA_TOKEN_FRACTION:-0.2}"
OPSA_FIXED_ADVANTAGE="${OPSA_FIXED_ADVANTAGE:--0.5}"
OPSA_ADVANTAGE_MIN="${OPSA_ADVANTAGE_MIN:--1.0}"
OPSA_ADVANTAGE_MAX="${OPSA_ADVANTAGE_MAX:--0.5}"

if [[ -f "$DATA_ROOT/smoke-8.jsonl" ]]; then
  PROMPT_DATA="${PROMPT_DATA:-$DATA_ROOT/smoke-8.jsonl}"
else
  PROMPT_DATA="${PROMPT_DATA:-$DATA_ROOT/dapo-math-17k.jsonl}"
fi

# ---------- Preconditions ----------
for path in "$SLIME_DIR" "$SGLANG_DIR" "$MEGATRON_DIR" "$HF_CKPT" "$MCORE_CKPT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
[[ -f "$PROMPT_DATA" ]] || { echo "ERROR: missing prompt data $PROMPT_DATA" >&2; exit 1; }

VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
NEEDED_GPUS="$((ACTOR_GPUS + ROLLOUT_GPUS))"

(( VISIBLE_GPUS >= NEEDED_GPUS )) || {
  echo "ERROR: visible GPUs=$VISIBLE_GPUS, but actor+rollout needs $NEEDED_GPUS" >&2
  exit 1
}

(( ACTOR_GPUS % TP_SIZE == 0 )) || {
  echo "ERROR: ACTOR_GPUS=$ACTOR_GPUS must be divisible by TP_SIZE=$TP_SIZE" >&2
  exit 1
}

case "$OPSA_MODE" in
  fixed|entropy) ;;
  *)
    echo "ERROR: OPSA_MODE must be fixed or entropy, got: $OPSA_MODE" >&2
    exit 1
    ;;
esac

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

# Fail before Ray startup if the runtime or OPSA port is broken.
python - <<'PY_PREFLIGHT'
import torch
import transformer_engine
import apex
import sglang
import megatron.core
import slime
from slime.backends.megatron_utils.opsa import compute_opsa

out = compute_opsa(
    [torch.tensor([-0.1, -3.0, -0.2, -2.0])],
    [torch.ones(4)],
    token_fraction=0.5,
    mode="fixed",
    fixed_advantage=-0.5,
)
assert out.loss_masks[0].tolist() == [0.0, 1.0, 0.0, 1.0]

print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count())
print("SGLang:", sglang.__version__)
print("TE:", transformer_engine.__version__)
print("OPSA preflight: OK")
PY_PREFLIGHT

# ---------- Logging ----------
mkdir -p "$OPSA_ROOT/logs"
LOG_FILE="$OPSA_ROOT/logs/qwen3-1.7b-opsa-${OPSA_MODE}-smoke-$(date +%Y%m%d-%H%M%S).log"

echo
echo "========== OPSA smoke =========="
echo "node:                 $(hostname)"
echo "visible GPUs:         $VISIBLE_GPUS"
echo "actor GPUs:           $ACTOR_GPUS"
echo "rollout GPUs:         $ROLLOUT_GPUS"
echo "TP size:              $TP_SIZE"
echo "OPSA mode:            $OPSA_MODE"
echo "token fraction:       $OPSA_TOKEN_FRACTION"
if [[ "$OPSA_MODE" == "fixed" ]]; then
  echo "fixed advantage:      $OPSA_FIXED_ADVANTAGE"
else
  echo "advantage range:      [$OPSA_ADVANTAGE_MIN, $OPSA_ADVANTAGE_MAX]"
fi
echo "HF checkpoint:        $HF_CKPT"
echo "Megatron checkpoint:  $MCORE_CKPT"
echo "prompt data:          $PROMPT_DATA"
echo "log:                  $LOG_FILE"
echo "================================"

cd "$SLIME_DIR"
source scripts/models/qwen3-1.7B.sh

# Actor initialization only. Because KL/reference losses are disabled,
# this does NOT create a reference-model training path.
CKPT_ARGS=(
  --hf-checkpoint "$HF_CKPT"
  --ref-load "$MCORE_CKPT"
)

# Teacher-free rollout: reward is deliberately zero; OPSA provides
# the token-level learning signal after current-actor logprob recomputation.
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

  --custom-rm-path slime.rollout.opsa.reward_func
  --custom-reward-post-process-path slime.rollout.opsa.post_process_rewards
)

# First smoke test: baseline OPSA only, no OPD teacher, no reference KL.
ALGORITHM_ARGS=(
  --advantage-estimator opsa
  --opsa-mode "$OPSA_MODE"
  --opsa-token-fraction "$OPSA_TOKEN_FRACTION"
  --kl-coef 0
  --entropy-coef 0
)

if [[ "$OPSA_MODE" == "fixed" ]]; then
  ALGORITHM_ARGS+=(
    --opsa-fixed-advantage "$OPSA_FIXED_ADVANTAGE"
  )
else
  ALGORITHM_ARGS+=(
    --opsa-advantage-min "$OPSA_ADVANTAGE_MIN"
    --opsa-advantage-max "$OPSA_ADVANTAGE_MAX"
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
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

# Sequence parallelism is only useful with TP > 1.
if (( TP_SIZE > 1 )); then
  PERF_ARGS+=(--sequence-parallel)
fi

# GH200/Hopper would otherwise select FA3. The current aarch64
# sgl_kernel wheel lacks flash_ops, so use the tested Triton backend.
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

# ---------- Ray ----------
cleanup() {
  set +e
  ray stop --force >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

ray stop --force >/dev/null 2>&1 || true
sleep 2

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

ray start \
  --head \
  --node-ip-address="$MASTER_ADDR" \
  --num-gpus="$VISIBLE_GPUS" \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265

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
ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${ALGORITHM_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
set +x

echo
echo "================================"
echo "OPSA SMOKE TEST FINISHED"
echo "Mode: $OPSA_MODE"
echo "Log:  $LOG_FILE"
echo
echo "Useful checks:"
echo "  grep -Ei 'opsa|selected_fraction|advantage_mean|nan|update_weights|succeeded' \"$LOG_FILE\" | tail -100"
echo "================================"
