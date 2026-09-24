#!/usr/bin/env bash
set -euo pipefail

salloc \
  -A ehpc-reg-2026r01-278-gpu \
  -p gpu \
  --gres=gpu:2 \
  --time=02:00:00
