#!/usr/bin/env bash
set -euo pipefail
# Exact OPSA-style base-model AIME24 sampling sanity check.
# Not suitable for the core tail study because top-k=20 and top-p=0.8 censor tail events.
for model in Qwen/Qwen3-1.7B Qwen/Qwen3-4B; do
  short_model="$(basename "${model}")"
  python run_topk_violation.py \
    --model "${model}" \
    --dataset aime24 \
    --samples-per-prompt 32 \
    --sampling-preset opsa \
    --k-values 1,2,5,10 \
    --primary-k 10 \
    --prefix-tokens 32,64,128 \
    --output-dir "outputs/opsa_sanity/${short_model}/aime24"
done
