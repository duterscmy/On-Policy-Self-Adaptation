#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
VARIANTS="${VARIANTS:-topk opsa fixed-negative}"
TOP_K="${TOP_K:-10}"
TOPK_ADVANTAGE="${TOPK_ADVANTAGE:--0.5}"

mkdir -p "$OUTPUT_ROOT"

for variant in $VARIANTS; do
   case "$variant" in
      topk)
         save_dir="${OUTPUT_ROOT}/topk_k${TOP_K}"
         extra=(--top-k "$TOP_K" --topk-advantage "$TOPK_ADVANTAGE")
         ;;
      opsa)
         save_dir="${OUTPUT_ROOT}/opsa"
         extra=(--fraction 0.2)
         ;;
      fixed-negative)
         save_dir="${OUTPUT_ROOT}/opsa_fixed_negative"
         extra=(--fraction 0.2)
         ;;
      *)
         echo "unknown variant: $variant" >&2
         exit 2
         ;;
   esac

   echo "============================================================"
   echo "Running ${variant} on 4 GPUs: 2 actor + 2 rollout"
   echo "Output: ${save_dir}"
   echo "============================================================"

   bash "${SCRIPT_DIR}/run_opsa.sh" \
      "$@" \
      --preset "$variant" \
      --actor-gpus 2 \
      --rollout-gpus 2 \
      --save-dir "$save_dir" \
      "${extra[@]}"
done
