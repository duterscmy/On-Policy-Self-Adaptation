#!/usr/bin/env bash
set -euo pipefail
python run_topk_violation.py \
  --model Qwen/Qwen3-1.7B \
  --dataset aime24 \
  --samples-per-prompt 2 \
  --limit 2 \
  --sampling-preset tail_analysis \
  --max-tokens 2048 \
  --k-values 1,5,10,20 \
  --primary-k 10 \
  --prefix-tokens 32,64 \
  --bootstrap-reps 100 \
  --output-dir outputs/smoke_qwen3_1.7b_aime24
