#!/bin/bash
#SBATCH --job-name="opsa_train"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/u6os/public/mingyu/opsa/slime
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -eo pipefail

# Slurm batch shells are non-interactive; initialize Conda explicitly.
source /home/u6os/cmy9797.u6os/miniconda3/etc/profile.d/conda.sh
conda activate opsa

# Runtime environment required by Transformer Engine / Triton on GH200 (aarch64).
# NCCL is header-only here on purpose: do NOT prepend the HPC SDK NCCL 2.19.3
# library to LD_LIBRARY_PATH because PyTorch 2.6.0+cu126 was built with NCCL 2.21.5.
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

export CPATH="$NCCL_INCLUDE:$CUDNN_ROOT/include:$MATHLIB_HOME/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CPATH"
export LIBRARY_PATH="$CUDNN_ROOT/lib:$MATHLIB_HOME/lib64:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDNN_ROOT/lib:$MATHLIB_HOME/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=9.0

# IMPORTANT: sbatch copies this script to /var/spool/slurmd/... before running it.
# Therefore do NOT derive project paths from BASH_SOURCE / dirname $0.
# Use stable absolute project paths instead. They can still be overridden via env vars.
MINGYU_ROOT="${MINGYU_ROOT:-/projects/u6os/public/mingyu}"
OPSA_ROOT="${OPSA_ROOT:-/projects/u6os/public/mingyu/opsa}"
SLIME_ROOT="${SLIME_ROOT:-/projects/u6os/public/mingyu/opsa/slime}"
OPSA_DATA_DIR="${OPSA_DATA_DIR:-/projects/u6os/public/mingyu/opsa/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/projects/u6os/public/mingyu/opsa/outputs}"
TRITON_HOME="${TRITON_HOME:-/projects/u6os/public/mingyu/.triton}"
export TRITON_HOME

MODEL="${MODEL:-qwen3-1.7b}"
PRESET="${PRESET:-topk}"
TOKEN_FRACTION="${TOKEN_FRACTION:-0.2}"
TOP_K="${TOP_K:-10}"
TOPK_ADVANTAGE="${TOPK_ADVANTAGE:--0.5}"
ACTOR_GPUS_OVERRIDE="${ACTOR_GPUS_OVERRIDE:-}"
ROLLOUT_GPUS_OVERRIDE="${ROLLOUT_GPUS_OVERRIDE:-}"
NUM_ROLLOUT_OVERRIDE="${NUM_ROLLOUT_OVERRIDE:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-${REF_LOAD:-}}"
RESUME_FROM="${RESUME_FROM:-}"
SAVE_DIR="${SAVE_DIR:-}"
PROMPT_DATA="${PROMPT_DATA:-${OPSA_DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-${OPSA_DATA_DIR}/aime-2024/aime-2024.jsonl}"
MEGATRON_PATH="${MEGATRON_PATH:-${MINGYU_ROOT}/models/Megatron-LM}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
# Job-specific ports avoid collisions when multiple Slurm jobs land on the same node.
RAY_PORT="${RAY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"
DASHBOARD_PORT="${DASHBOARD_PORT:-$((30000 + ${SLURM_JOB_ID:-0} % 10000))}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_LOG_ALL_METRICS="${WANDB_LOG_ALL_METRICS:-false}"
WANDB_OPEN_METRICS="${WANDB_OPEN_METRICS:-false}"
LIGHT_CHECKPOINT=false
DRY_RUN=false

usage() {
   cat <<'EOF'
Usage:
  sbatch examples/opsa/run_opsa.sh [options]

Method:
  --model NAME                qwen3-1.7b, qwen3-4b, or qwen3.5-9b
  --preset NAME               opsa, fixed-negative, fixed-positive, or topk
  --fraction FLOAT            Lowest-token fraction in (0, 1] (default: 0.2)
  --top-k INTEGER             Top-K support for --preset topk (default: 10)
  --topk-advantage FLOAT      Negative Top-K suppression advantage (default: -0.5)
  --steps INTEGER             Override the model preset's training steps

Paths:
  --hf-checkpoint PATH        Hugging Face checkpoint used by rollout
  --actor-checkpoint PATH     Megatron checkpoint used to initialize the actor
  --resume-from PATH          Resume a prior full training checkpoint
  --save-dir PATH             Output checkpoint directory
  --prompt-data FILE          Training JSON/JSONL file
  --eval-data FILE            Evaluation JSON/JSONL file
  --megatron-path DIR         Megatron-LM checkout

Runtime:
  --actor-gpus INTEGER        Override actor GPU count from the model preset
  --rollout-gpus INTEGER      Override rollout GPU count from the model preset
  --ray-address HOST:PORT     Reuse an existing Ray GCS address
  --ray-port PORT             Local Ray GCS port (default: 6379)
  --dashboard-port PORT       Local Ray dashboard port (default: 8265)
  --light-checkpoint          Omit optimizer and RNG state (not resumable)
  --dry-run                   Print the resolved command without checking paths
  -h, --help                  Show this help

W&B (disabled unless a project is provided):
  --wandb-project NAME        Enable W&B and log to this project
  --wandb-group NAME          Group/run name (default includes model, preset, fraction)
  --wandb-team NAME           W&B entity/team
  --wandb-mode MODE           online, offline, or disabled
  --wandb-dir PATH            Directory for local W&B files
  --wandb-log-all-metrics     Log all Slime metrics instead of the compact set
  --wandb-open-metrics        Add SGLang OpenMetrics to an online W&B run

The corresponding uppercase environment variables may be used. Boolean W&B
environment variables accept true/false, 1/0, yes/no, or on/off. Command-line
values take precedence. API keys are never accepted as launcher arguments or
placed in printed commands; inject WANDB_API_KEY through the environment or a
cluster secret.
EOF
}

die() {
   echo "error: $*" >&2
   exit 2
}

require_value() {
   if [ "$#" -lt 2 ] || [ -z "$2" ]; then
      die "$1 requires a value"
   fi
}

normalize_boolean() {
   local name="$1"
   local value="$2"
   case "$value" in
      1|true|TRUE|yes|YES|on|ON) echo true ;;
      0|false|FALSE|no|NO|off|OFF|"") echo false ;;
      *) die "$name must be a boolean (true/false, 1/0, yes/no, or on/off)" ;;
   esac
}

WANDB_LOG_ALL_METRICS="$(normalize_boolean WANDB_LOG_ALL_METRICS "$WANDB_LOG_ALL_METRICS")"
WANDB_OPEN_METRICS="$(normalize_boolean WANDB_OPEN_METRICS "$WANDB_OPEN_METRICS")"

while [ "$#" -gt 0 ]; do
   case "$1" in
      --model)
         require_value "$@"
         MODEL="$2"
         shift 2
         ;;
      --preset)
         require_value "$@"
         PRESET="$2"
         shift 2
         ;;
      --fraction)
         require_value "$@"
         TOKEN_FRACTION="$2"
         shift 2
         ;;
      --top-k)
         require_value "$@"
         TOP_K="$2"
         shift 2
         ;;
      --topk-advantage)
         require_value "$@"
         TOPK_ADVANTAGE="$2"
         shift 2
         ;;
      --steps)
         require_value "$@"
         NUM_ROLLOUT_OVERRIDE="$2"
         shift 2
         ;;
      --hf-checkpoint)
         require_value "$@"
         HF_CHECKPOINT="$2"
         shift 2
         ;;
      --actor-checkpoint)
         require_value "$@"
         ACTOR_CHECKPOINT="$2"
         shift 2
         ;;
      --resume-from)
         require_value "$@"
         RESUME_FROM="$2"
         shift 2
         ;;
      --save-dir)
         require_value "$@"
         SAVE_DIR="$2"
         shift 2
         ;;
      --prompt-data)
         require_value "$@"
         PROMPT_DATA="$2"
         shift 2
         ;;
      --eval-data)
         require_value "$@"
         EVAL_DATA="$2"
         shift 2
         ;;
      --megatron-path)
         require_value "$@"
         MEGATRON_PATH="$2"
         shift 2
         ;;
      --actor-gpus)
         require_value "$@"
         ACTOR_GPUS_OVERRIDE="$2"
         shift 2
         ;;
      --rollout-gpus)
         require_value "$@"
         ROLLOUT_GPUS_OVERRIDE="$2"
         shift 2
         ;;
      --ray-address)
         require_value "$@"
         RAY_ADDRESS="$2"
         shift 2
         ;;
      --ray-port)
         require_value "$@"
         RAY_PORT="$2"
         shift 2
         ;;
      --dashboard-port)
         require_value "$@"
         DASHBOARD_PORT="$2"
         shift 2
         ;;
      --wandb-project)
         require_value "$@"
         WANDB_PROJECT="$2"
         shift 2
         ;;
      --wandb-group)
         require_value "$@"
         WANDB_GROUP="$2"
         shift 2
         ;;
      --wandb-team)
         require_value "$@"
         WANDB_TEAM="$2"
         shift 2
         ;;
      --wandb-mode)
         require_value "$@"
         WANDB_MODE="$2"
         shift 2
         ;;
      --wandb-dir)
         require_value "$@"
         WANDB_DIR="$2"
         shift 2
         ;;
      --wandb-log-all-metrics)
         WANDB_LOG_ALL_METRICS=true
         shift
         ;;
      --wandb-open-metrics)
         WANDB_OPEN_METRICS=true
         shift
         ;;
      --light-checkpoint)
         LIGHT_CHECKPOINT=true
         shift
         ;;
      --dry-run)
         DRY_RUN=true
         shift
         ;;
      -h|--help)
         usage
         exit 0
         ;;
      *)
         die "unknown option: $1"
         ;;
   esac
done

case "$MODEL" in
   qwen3-1.7b|qwen3-4b|qwen3.5-9b) ;;
   *) die "unsupported model '$MODEL'" ;;
esac

case "$PRESET" in
   opsa|fixed-negative|fixed-positive|topk) ;;
   *) die "unsupported preset '$PRESET'" ;;
esac

# Qwen3-1.7B defaults: 2 actor GPUs + 2 rollout GPUs, plus local checkpoints.
# Other models keep their model-file GPU defaults unless explicitly overridden.
if [ "$MODEL" = qwen3-1.7b ]; then
   ACTOR_GPUS_OVERRIDE="${ACTOR_GPUS_OVERRIDE:-2}"
   ROLLOUT_GPUS_OVERRIDE="${ROLLOUT_GPUS_OVERRIDE:-2}"
   HF_CHECKPOINT="${HF_CHECKPOINT:-${MINGYU_ROOT}/models/Qwen3-1.7B}"
   ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-${MINGYU_ROOT}/models/qwen3-1.7b-megatron}"
fi

# Automatically create a unique output directory when --save-dir is omitted.
if [ -z "$SAVE_DIR" ]; then
   RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
   case "$PRESET" in
      topk) EXP_NAME="topk-k${TOP_K}-advneg${TOPK_ADVANTAGE#-}" ;;
      opsa) EXP_NAME="opsa-lowest${TOKEN_FRACTION}" ;;
      fixed-negative) EXP_NAME="fixed-negative-lowest${TOKEN_FRACTION}" ;;
      fixed-positive) EXP_NAME="fixed-positive-lowest${TOKEN_FRACTION}" ;;
   esac
   SAVE_DIR="${OUTPUT_ROOT}/${MODEL}/${EXP_NAME}-${RUN_STAMP}"
fi

if ! [[ "$TOP_K" =~ ^[1-9][0-9]*$ ]]; then
   die "--top-k/TOP_K must be a positive integer"
fi
if [ "$PRESET" = topk ] && ! awk -v value="$TOPK_ADVANTAGE" 'BEGIN { exit !(value < 0) }'; then
   die "--topk-advantage/TOPK_ADVANTAGE must be negative"
fi

if ! [[ "$TOKEN_FRACTION" =~ ^(0([.][0-9]+)?|1([.]0*)?)$ ]]; then
   die "--fraction must be a number in (0, 1]"
fi
if ! awk -v fraction="$TOKEN_FRACTION" 'BEGIN { exit !(fraction > 0 && fraction <= 1) }'; then
   die "--fraction must be a number in (0, 1]"
fi
if [ -n "$NUM_ROLLOUT_OVERRIDE" ] && ! [[ "$NUM_ROLLOUT_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
   die "--steps/NUM_ROLLOUT_OVERRIDE must be a positive integer"
fi

validate_port() {
   local option="$1"
   local port="$2"
   case "$port" in
      ""|*[!0-9]*) die "$option must be an integer from 1 to 65535" ;;
   esac
   if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
      die "$option must be an integer from 1 to 65535"
   fi
}

port_is_free() {
   python3 -c 'import socket, sys; sock = socket.socket(); sock.bind(("0.0.0.0", int(sys.argv[1]))); sock.close()' "$1" 2>/dev/null
}

validate_port "--ray-port" "$RAY_PORT"
validate_port "--dashboard-port" "$DASHBOARD_PORT"
if [ "$RAY_PORT" -eq "$DASHBOARD_PORT" ]; then
   die "--ray-port and --dashboard-port must use different ports"
fi
if [ "$LIGHT_CHECKPOINT" = true ] && [ -n "$RESUME_FROM" ]; then
   die "--light-checkpoint cannot be combined with --resume-from"
fi
if [ -n "$WANDB_PROJECT" ]; then
   case "$WANDB_MODE" in
      ""|online|offline|disabled) ;;
      *) die "--wandb-mode must be online, offline, or disabled" ;;
   esac
   if [ "$WANDB_OPEN_METRICS" = true ] && [ "$WANDB_MODE" != "" ] && [ "$WANDB_MODE" != online ]; then
      die "--wandb-open-metrics requires online W&B mode"
   fi
fi

source "${SLIME_ROOT}/examples/opsa/models/${MODEL}.sh"
MODEL_CONFIG="${SLIME_ROOT}/${MODEL_CONFIG_RELATIVE}"
if [ ! -f "$MODEL_CONFIG" ]; then
   die "model definition not found: $MODEL_CONFIG"
fi
source "$MODEL_CONFIG"
if [ -n "$ACTOR_GPUS_OVERRIDE" ]; then
   ACTOR_GPUS="$ACTOR_GPUS_OVERRIDE"
fi
if [ -n "$ROLLOUT_GPUS_OVERRIDE" ]; then
   ROLLOUT_GPUS="$ROLLOUT_GPUS_OVERRIDE"
fi
if [ -n "$NUM_ROLLOUT_OVERRIDE" ]; then
   NUM_ROLLOUT="$NUM_ROLLOUT_OVERRIDE"
fi

TOTAL_GPUS=$((ACTOR_GPUS + ROLLOUT_GPUS))
if [ $((ACTOR_GPUS % TENSOR_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
   die "actor GPUs must be divisible by tensor model parallel size"
fi
if [ $((ROLLOUT_GPUS % ROLLOUT_GPUS_PER_ENGINE)) -ne 0 ]; then
   die "rollout GPUs must be divisible by rollout GPUs per engine"
fi

if [ "$DRY_RUN" = true ]; then
   HF_CHECKPOINT="${HF_CHECKPOINT:-<HF_CHECKPOINT>}"
   ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-<ACTOR_MEGATRON_CHECKPOINT>}"
   SAVE_DIR="${SAVE_DIR:-<SAVE_DIR>}"
   PROMPT_DATA="${PROMPT_DATA:-<TRAIN_DATA>}"
   EVAL_DATA="${EVAL_DATA:-<EVAL_DATA>}"
   MEGATRON_PATH="${MEGATRON_PATH:-<MEGATRON_LM>}"
else
   [ -n "$HF_CHECKPOINT" ] || die "--hf-checkpoint is required"
   [ -n "$ACTOR_CHECKPOINT" ] || die "--actor-checkpoint is required"
   [ -n "$SAVE_DIR" ] || die "--save-dir is required"
   [ -n "$PROMPT_DATA" ] || die "--prompt-data is required"
   [ -n "$EVAL_DATA" ] || die "--eval-data is required"
   [ -n "$MEGATRON_PATH" ] || die "--megatron-path is required"

   [ -d "$HF_CHECKPOINT" ] || die "Hugging Face checkpoint is not a directory: $HF_CHECKPOINT"
   [ -d "$ACTOR_CHECKPOINT" ] || die "actor checkpoint is not a directory: $ACTOR_CHECKPOINT"
   [ -f "$PROMPT_DATA" ] || die "training data is not a file: $PROMPT_DATA"
   [ -f "$EVAL_DATA" ] || die "evaluation data is not a file: $EVAL_DATA"
   [ -d "$MEGATRON_PATH" ] || die "Megatron-LM path is not a directory: $MEGATRON_PATH"
   if [ -n "$RESUME_FROM" ] && [ ! -d "$RESUME_FROM" ]; then
      die "resume checkpoint is not a directory: $RESUME_FROM"
   fi

   command -v python3 >/dev/null 2>&1 || die "python3 is required"
   command -v ray >/dev/null 2>&1 || die "ray is required"

   if [ -z "$RAY_ADDRESS" ]; then
      command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required when starting local Ray"
      if ray status >/dev/null 2>&1; then
         die "a local Ray cluster is already running; pass --ray-address to reuse it"
      fi
      GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader)" ||
         die "failed to query local GPUs"
      DETECTED_GPUS=0
      while IFS= read -r gpu_index; do
         if [ -n "$gpu_index" ]; then
            DETECTED_GPUS=$((DETECTED_GPUS + 1))
         fi
      done <<< "$GPU_LIST"
      if [ "$DETECTED_GPUS" -lt "$TOTAL_GPUS" ]; then
         die "$MODEL requires $TOTAL_GPUS local GPUs, but only $DETECTED_GPUS were detected"
      fi
      if ! port_is_free "$RAY_PORT"; then
         die "local Ray GCS port is already in use: $RAY_PORT"
      fi
      if ! port_is_free "$DASHBOARD_PORT"; then
         die "local Ray dashboard port is already in use: $DASHBOARD_PORT"
      fi
   fi
   mkdir -p "$SAVE_DIR"
fi

case "$PRESET" in
   opsa)
      OPSA_ARGS=(
         --opsa-mode entropy
         --opsa-token-fraction "$TOKEN_FRACTION"
         --opsa-advantage-min -1.0
         --opsa-advantage-max -0.5
      )
      ;;
   fixed-negative)
      OPSA_ARGS=(
         --opsa-mode fixed
         --opsa-token-fraction "$TOKEN_FRACTION"
         --opsa-fixed-advantage -0.5
      )
      ;;
   fixed-positive)
      OPSA_ARGS=(
         --opsa-mode fixed
         --opsa-token-fraction "$TOKEN_FRACTION"
         --opsa-fixed-advantage 0.2
      )
      ;;
   topk)
      OPSA_ARGS=(
         --opsa-mode topk
         --opsa-top-k "$TOP_K"
         --opsa-fixed-advantage "$TOPK_ADVANTAGE"
      )
      ;;
esac

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$ACTOR_CHECKPOINT"
   --save "$SAVE_DIR"
   --save-interval 20
)
if [ -n "$RESUME_FROM" ]; then
   CKPT_ARGS+=(--load "$RESUME_FROM")
fi
if [ "$LIGHT_CHECKPOINT" = true ]; then
   CKPT_ARGS+=(--no-save-optim --no-save-rng --no-load-optim --no-load-rng)
fi

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt
   --apply-chat-template
   --disable-thinking
   --rollout-shuffle
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size 64
   --n-samples-per-prompt 1
   --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
   --rollout-temperature 1
   --num-steps-per-rollout 1
   --global-batch-size 64
   --balance-data
   --custom-rm-path slime.rollout.opsa.reward_func
   --custom-reward-post-process-path slime.rollout.opsa.post_process_rewards
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "$EVAL_DATA"
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt 4
   --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
   --eval-top-p 0.8
   --eval-temperature 0.7
   --eval-top-k 20
   --eval-rm-type math
   --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
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

ALGORITHM_ARGS=(
   --advantage-estimator opsa
   "${OPSA_ARGS[@]}"
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
if [ "$OPTIMIZER_CPU_OFFLOAD" = true ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
   --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
   --sglang-attention-backend triton
)

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
   if [ "$PRESET" = topk ]; then
      WANDB_GROUP="${WANDB_GROUP:-opsa-${MODEL}-topk${TOP_K}}"
   else
      fraction_percentage="$(awk -v value="$TOKEN_FRACTION" 'BEGIN { printf "%g", value * 100 }')"
      fraction_percentage="${fraction_percentage//./p}"
      WANDB_GROUP="${WANDB_GROUP:-opsa-${MODEL}-${PRESET}-lowest${fraction_percentage}}"
   fi
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-group "$WANDB_GROUP"
      --disable-wandb-random-suffix
   )
   if [ -n "$WANDB_TEAM" ]; then
      WANDB_ARGS+=(--wandb-team "$WANDB_TEAM")
   fi
   if [ -n "$WANDB_MODE" ]; then
      WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
   fi
   if [ -n "$WANDB_DIR" ]; then
      WANDB_ARGS+=(--wandb-dir "$WANDB_DIR")
   fi
   if [ "$WANDB_LOG_ALL_METRICS" = true ]; then
      WANDB_ARGS+=(--wandb-log-all-metrics)
   fi
   if [ "$WANDB_OPEN_METRICS" = true ]; then
      WANDB_ARGS+=(--wandb-open-metrics)
   fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

TRAIN_COMMAND=(
   "${CONDA_PREFIX}/bin/python" "${SLIME_ROOT}/train.py"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "$ACTOR_GPUS"
   --rollout-num-gpus "$ROLLOUT_GPUS"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${ALGORITHM_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

# Fail early inside the allocated Slurm node if the CUDA stack is not visible.
if [ "$DRY_RUN" = false ]; then
python3 - <<'PY_PREFLIGHT'
import torch, triton, transformer_engine, transformer_engine.pytorch, apex
import fused_weight_gradient_mlp_cuda
print("[preflight] torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("[preflight] GPU:", torch.cuda.get_device_name(0))
print("[preflight] Triton:", triton.__version__)
print("[preflight] TE:", transformer_engine.__version__)
print("[preflight] Apex fused_weight_gradient_mlp_cuda: OK")
print("[preflight] visible GPUs:", torch.cuda.device_count())
PY_PREFLIGHT
fi

HAS_NVLINK=0
if [ "$DRY_RUN" = false ] && [ -z "$RAY_ADDRESS" ]; then
   GPU_TOPOLOGY="$(nvidia-smi topo -m 2>/dev/null || true)"
   if [[ "$GPU_TOPOLOGY" == *"NV"* ]]; then
      HAS_NVLINK=1
   fi
fi

RUNTIME_PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export RUNTIME_PYTHONPATH HAS_NVLINK
RUNTIME_ENV_JSON="$(
python3 - <<'PY_RUNTIME_ENV'
import json, os
keys = [
    "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH",
    "CUDA_HOME", "MATHLIB_HOME", "NCCL_INCLUDE", "CUDNN_ROOT",
    "TRITON_HOME", "TORCH_CUDA_ARCH_LIST",
]
env = {k: os.environ[k] for k in keys if k in os.environ}
env.update({
    "PYTHONPATH": os.environ["RUNTIME_PYTHONPATH"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"],
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
})
print(json.dumps({"env_vars": env}))
PY_RUNTIME_ENV
)"

echo "Model:               $MODEL_DISPLAY_NAME"
echo "Preset:              $PRESET"
if [ "$PRESET" = topk ]; then
   echo "Top-K suppression:   K=${TOP_K}, advantage=${TOPK_ADVANTAGE}"
else
   echo "Lowest fraction:     $TOKEN_FRACTION"
fi
echo "Actor/Rollout GPUs:  ${ACTOR_GPUS}/${ROLLOUT_GPUS}"
echo "Training TP:         $TENSOR_MODEL_PARALLEL_SIZE"
echo "Training steps:      $NUM_ROLLOUT"
echo "Rollout/Eval length: ${ROLLOUT_MAX_RESPONSE_LEN}/${EVAL_MAX_RESPONSE_LEN}"
echo "Checkpoint mode:     $([ "$LIGHT_CHECKPOINT" = true ] && echo light || echo resumable)"
if [ -n "$WANDB_PROJECT" ]; then
   echo "W&B:                 project=$WANDB_PROJECT group=$WANDB_GROUP mode=${WANDB_MODE:-online}"
   echo "W&B metrics:         $([ "$WANDB_LOG_ALL_METRICS" = true ] && echo all || echo compact)"
   echo "W&B OpenMetrics:     $([ "$WANDB_OPEN_METRICS" = true ] && echo enabled || echo disabled)"
else
   echo "W&B:                 disabled"
fi

# Single-node Slurm mode: let the Python driver create its own local Ray
# runtime with ray.init().  Do NOT start `ray start` separately and then attach
# to it.  On this cluster the attach path has repeatedly hung inside ray.init().
# Starting Ray from the driver removes the GCS/raylet attachment handshake,
# Ray Jobs API, dashboard, and repository packaging layers.

if [ -n "$RAY_ADDRESS" ]; then
   die "this single-node Slurm launcher intentionally does not reuse an external Ray cluster; unset RAY_ADDRESS"
fi

NODE_IP="$(
python3 - <<'PY_NODE_IP'
import socket
name = socket.gethostname()
ips = []
try:
    ips.extend(socket.gethostbyname_ex(name)[2])
except OSError:
    pass
ips = [ip for ip in ips if ":" not in ip and not ip.startswith("127.")]
if not ips:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 53))
        ips.append(s.getsockname()[0])
    finally:
        s.close()
if not ips:
    raise SystemExit("could not determine the Slurm node IPv4 address")
print(ips[0])
PY_NODE_IP
)"
[ -n "$NODE_IP" ] || die "failed to determine Slurm node IP"

RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-16}"
RAY_NUM_GPUS="$TOTAL_GPUS"

# Keep Ray's session/socket files job-local.  This avoids stale state in
# /tmp/ray from an earlier Slurm job on the same compute node.
RAY_TMPDIR="${TMPDIR:-/tmp}/ray-opsa-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"

export PYTHONPATH="$RUNTIME_PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="$HAS_NVLINK"
export SGLANG_DISABLE_CUDNN_CHECK=1

export RAY_NODE_IP="$NODE_IP"
export RAY_NUM_CPUS
export RAY_NUM_GPUS
export RAY_TMPDIR

echo "Ray launch mode:     local ray.init() inside driver"
echo "Ray node IP:         $RAY_NODE_IP"
echo "Ray CPUs/GPUs:       ${RAY_NUM_CPUS}/${RAY_NUM_GPUS}"
echo "Ray temp dir:        $RAY_TMPDIR"

if [ "$DRY_RUN" = true ]; then
   printf '\n[dry-run] train:'
   printf ' %q' "${TRAIN_COMMAND[@]}"
   printf '\n'
   exit 0
fi

cd "$SLIME_ROOT"

echo "Starting Slime driver; Python will create the local Ray runtime..."

"${CONDA_PREFIX}/bin/python" -u - "$SLIME_ROOT/train.py" "${TRAIN_COMMAND[@]:2}" <<'PY_DIRECT_DRIVER'
import os
import runpy
import signal
import sys

# Keep BLAS/OpenMP from creating huge thread pools during Ray startup on HPC
# nodes.  Slime/Ray actors can still use their explicitly allocated CPUs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import ray

train_py = sys.argv[1]
train_args = sys.argv[2:]

node_ip = os.environ["RAY_NODE_IP"]
num_cpus = int(os.environ["RAY_NUM_CPUS"])
num_gpus = int(os.environ["RAY_NUM_GPUS"])
temp_dir = os.environ["RAY_TMPDIR"]

print(
    f"[ray] starting local runtime: node_ip={node_ip}, "
    f"cpus={num_cpus}, gpus={num_gpus}, temp_dir={temp_dir}",
    flush=True,
)

def _ray_init_timeout(signum, frame):
    raise TimeoutError(
        "local ray.init() did not complete within 120 seconds; "
        f"inspect {temp_dir}/session_latest/logs/"
    )

signal.signal(signal.SIGALRM, _ray_init_timeout)
signal.alarm(120)
try:
    ctx = ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        include_dashboard=False,
        _node_ip_address=node_ip,
        _temp_dir=temp_dir,
        log_to_driver=True,
    )
finally:
    signal.alarm(0)

print("[ray] local runtime started", flush=True)
print("[ray] address:", ctx.address_info.get("address"), flush=True)
print("[ray] cluster resources:", ray.cluster_resources(), flush=True)

sys.argv = [train_py, *train_args]
try:
    runpy.run_path(train_py, run_name="__main__")
finally:
    print("[ray] shutting down local runtime", flush=True)
    ray.shutdown()
PY_DIRECT_DRIVER
