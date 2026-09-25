#!/usr/bin/env bash
set -euo pipefail

# Four-GPU Qwen3-1.7B OPD/OPSA comparison on Arrhenius.
# Expected allocation: 2 actor GPUs + 2 rollout GPUs on one GH200 node.

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
export PROMPT_DATA="${PROMPT_DATA:-$OPSA_ROOT/data/dapo-math-17k/dapo-math-17k.jsonl}"
export EVAL_DATA="${EVAL_DATA:-$OPSA_ROOT/data/aime-2024/aime-2024.jsonl}"

export CUDA_HOME="${CUDA_HOME:-/software/sse2/el9_gh200/easybuild/prefix/software/CUDA/12.9.1}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_ROOT="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn"
export NCCL_HOME="/software/sse2/el9_gh200/easybuild/prefix/software/NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1"
export GCC14_LIB="/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64"
export TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="$GCC14_LIB:$CUDNN_ROOT/lib:$NCCL_HOME/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SGLANG_DIR/python:${PYTHONPATH:-}"

METHOD="${METHOD:-opd}"
OPD_LOSS_TYPE="${OPD_LOSS_TYPE:-vanilla}"
OPD_TOKEN_FILTER="${OPD_TOKEN_FILTER:-all}"
OPD_HIGH_CONF_THRESHOLD="${OPD_HIGH_CONF_THRESHOLD:-0.5}"
OPD_BOTTOM_FRACTION="${OPD_BOTTOM_FRACTION:-0.2}"
OPD_GEOMETRY_ALPHA="${OPD_GEOMETRY_ALPHA:-0.5}"
OPSA_MODE="${OPSA_MODE:-fixed}"
OPSA_TOKEN_FRACTION="${OPSA_TOKEN_FRACTION:-0.2}"
OPSA_FIXED_ADVANTAGE="${OPSA_FIXED_ADVANTAGE:--0.5}"

ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES="${N_SAMPLES:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
RETAIN_BEST_AND_LAST="${RETAIN_BEST_AND_LAST:-1}"
RETENTION_METRIC="${RETENTION_METRIC:-eval/aime}"
SEED="${SEED:-1234}"

case "$METHOD" in
  opd)
    DEFAULT_RUN_NAME="opd-${OPD_LOSS_TYPE}-${OPD_TOKEN_FILTER}"
    ;;
  opsa)
    DEFAULT_RUN_NAME="opsa-${OPSA_MODE}-bottom${OPSA_TOKEN_FRACTION}"
    ;;
  *)
    echo "ERROR: METHOD must be opd or opsa, got $METHOD" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
SAVE_DIR="${SAVE_DIR:-$OPSA_ROOT/outputs/qwen3-1.7b-$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$OPSA_ROOT/logs/qwen3-1.7b-$RUN_NAME-${SLURM_JOB_ID:-manual}.log}"

for path in "$SLIME_DIR" "$SGLANG_DIR" "$MEGATRON_DIR" "$HF_CKPT" "$MCORE_CKPT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
for file in "$PROMPT_DATA" "$EVAL_DATA"; do
  [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done

VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
NEEDED_GPUS="$((ACTOR_GPUS + ROLLOUT_GPUS))"
(( VISIBLE_GPUS >= NEEDED_GPUS )) || {
  echo "ERROR: visible GPUs=$VISIBLE_GPUS, need $NEEDED_GPUS" >&2
  exit 1
}
(( ACTOR_GPUS % TP_SIZE == 0 )) || {
  echo "ERROR: ACTOR_GPUS must be divisible by TP_SIZE" >&2
  exit 1
}

mkdir -p "$OPSA_ROOT/logs" "$SAVE_DIR"

echo "========== Qwen3-1.7B OPD/OPSA study =========="
echo "host/job:              $(hostname) / ${SLURM_JOB_ID:-none}"
echo "method:                $METHOD"
echo "run:                   $RUN_NAME"
echo "OPD objective/filter:  $OPD_LOSS_TYPE / $OPD_TOKEN_FILTER"
echo "OPSA mode/fraction:    $OPSA_MODE / $OPSA_TOKEN_FRACTION"
echo "actor/rollout GPUs:    $ACTOR_GPUS / $ROLLOUT_GPUS"
echo "rollouts:              $NUM_ROLLOUT"
echo "batch:                 $ROLLOUT_BATCH_SIZE x $N_SAMPLES = $GLOBAL_BATCH_SIZE"
echo "response/eval length:  $MAX_RESPONSE_LEN / $EVAL_MAX_RESPONSE_LEN"
echo "eval samples/prompt:    $N_SAMPLES_PER_EVAL_PROMPT"
echo "save dir:              $SAVE_DIR"
echo "log:                   $LOG_FILE"
echo "================================================"

python - <<'PY_PREFLIGHT'
import torch
import transformer_engine
import apex
import sglang
import megatron.core
import slime

print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count(), "SGLang:", sglang.__version__)
print("runtime preflight: OK")
PY_PREFLIGHT

cd "$SLIME_DIR"
source scripts/models/qwen3-1.7B.sh

CKPT_ARGS=(
  --hf-checkpoint "$HF_CKPT"
  --ref-load "$MCORE_CKPT"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
  --no-save-optim
  --no-save-rng
)

ROLLOUT_ARGS=(
  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --disable-thinking
  --rollout-shuffle
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$MAX_RESPONSE_LEN"
  --rollout-temperature 1.0
  --num-steps-per-rollout 1
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)

EVAL_ARGS=(
  --eval-interval "$EVAL_INTERVAL"
  --eval-prompt-data aime "$EVAL_DATA"
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT"
  --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
  --eval-top-p 0.8
  --eval-temperature 0.7
  --eval-top-k 20
  --eval-rm-type math
  --log-passrate
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
if (( TP_SIZE > 1 )); then
  PERF_ARGS+=(--sequence-parallel)
fi

if [[ "$METHOD" == "opd" ]]; then
  ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type megatron
    --opd-kl-coef 1.0
    --opd-loss-type "$OPD_LOSS_TYPE"
    --opd-token-filter "$OPD_TOKEN_FILTER"
    --opd-high-conf-threshold "$OPD_HIGH_CONF_THRESHOLD"
    --opd-bottom-fraction "$OPD_BOTTOM_FRACTION"
    --opd-geometry-alpha "$OPD_GEOMETRY_ALPHA"
    --opd-teacher-load "$MCORE_CKPT"
    --kl-coef 0
    --entropy-coef 0
  )
  RM_ARGS=(--rm-type math)
else
  ALGORITHM_ARGS=(
    --advantage-estimator opsa
    --opsa-mode "$OPSA_MODE"
    --opsa-token-fraction "$OPSA_TOKEN_FRACTION"
    --kl-coef 0
    --entropy-coef 0
  )
  if [[ "$OPSA_MODE" == "fixed" ]]; then
    ALGORITHM_ARGS+=(--opsa-fixed-advantage "$OPSA_FIXED_ADVANTAGE")
  fi
  RM_ARGS=(
    --custom-rm-path slime.rollout.opsa.reward_func
    --custom-reward-post-process-path slime.rollout.opsa.post_process_rewards
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

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static 0.65
  --sglang-attention-backend triton
  --sglang-enable-deterministic-inference
)

MISC_ARGS=(
  --seed "$SEED"
  --log-interval 1
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
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"

ray start \
  --head \
  --node-ip-address="$MASTER_ADDR" \
  --num-gpus="$VISIBLE_GPUS" \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="$DASHBOARD_PORT"

RUNTIME_ENV_JSON="$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_HOME":"%s","CUDA_PATH":"%s","LD_LIBRARY_PATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","TOKENIZERS_PARALLELISM":"false"}}' \
  "$PYTHONPATH" "$CUDA_HOME" "$CUDA_PATH" "$LD_LIBRARY_PATH")"

set -x
ray job submit \
  --address="http://127.0.0.1:$DASHBOARD_PORT" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  --num-gpus-per-node "$VISIBLE_GPUS" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${ALGORITHM_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
set +x

if [[ "$RETAIN_BEST_AND_LAST" == "1" ]]; then
  python3 scripts/retain_best_checkpoint.py \
    --save-dir "$SAVE_DIR" \
    --log-file "$LOG_FILE" \
    --metric "$RETENTION_METRIC" \
    --apply
fi

echo "EXPERIMENT FINISHED: $RUN_NAME"
echo "Log: $LOG_FILE"
