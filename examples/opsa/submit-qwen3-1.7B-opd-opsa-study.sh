#!/usr/bin/env bash
set -euo pipefail

# Submit independent four-GPU comparison jobs. Slurm may run as many in
# parallel as the account's fair-share and available nodes permit.

OPSA_ROOT="${OPSA_ROOT:-/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa}"
SLIME_DIR="${SLIME_DIR:-$OPSA_ROOT/slime-upstream}"
RUNNER="$SLIME_DIR/examples/opsa/run-qwen3-1.7B-opd-opsa-study.sh"
ACCOUNT="${SLURM_ACCOUNT:-ehpc-reg-2026r01-278-gpu}"
PARTITION="${SLURM_PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
INCLUDE_VANILLA="${INCLUDE_VANILLA:-1}"

[[ -x "$RUNNER" ]] || { echo "ERROR: runner is not executable: $RUNNER" >&2; exit 1; }
mkdir -p "$OPSA_ROOT/logs"

submit_condition() {
  local name="$1"
  local exports="$2"
  local job_id
  job_id="$(
    sbatch --parsable \
      -A "$ACCOUNT" \
      -p "$PARTITION" \
      --nodes=1 \
      --gres=gpu:4 \
      --time="$TIME_LIMIT" \
      --job-name="$name" \
      --output="$OPSA_ROOT/logs/study-%x-%j.log" \
      --export="ALL,NUM_ROLLOUT=200,EVAL_INTERVAL=20,SAVE_INTERVAL=20,RETAIN_BEST_AND_LAST=1,$exports" \
      "$RUNNER"
  )"
  echo "$name $job_id"
}

# Set INCLUDE_VANILLA=0 when a compatible vanilla baseline is already running.
if [[ "$INCLUDE_VANILLA" == "1" ]]; then
  submit_condition "opd-vanilla" \
    "RUN_NAME=opd-vanilla-all,METHOD=opd,OPD_LOSS_TYPE=vanilla,OPD_TOKEN_FILTER=all"
fi
submit_condition "opsa-fixed" \
  "RUN_NAME=opsa-fixed-bottom20,METHOD=opsa,OPSA_MODE=fixed,OPSA_TOKEN_FRACTION=0.2"
submit_condition "opd-gc" \
  "RUN_NAME=opd-gc-all,METHOD=opd,OPD_LOSS_TYPE=geometry_corrected,OPD_TOKEN_FILTER=all"
submit_condition "opd-hc-vanilla" \
  "RUN_NAME=opd-vanilla-highconf,METHOD=opd,OPD_LOSS_TYPE=vanilla,OPD_TOKEN_FILTER=high_conf"
submit_condition "opd-hc-gc" \
  "RUN_NAME=opd-gc-highconf,METHOD=opd,OPD_LOSS_TYPE=geometry_corrected,OPD_TOKEN_FILTER=high_conf"
submit_condition "opd-bernoulli" \
  "RUN_NAME=opd-bernoulli-all,METHOD=opd,OPD_LOSS_TYPE=bernoulli,OPD_TOKEN_FILTER=all"
submit_condition "opd-weighted" \
  "RUN_NAME=opd-weighted-a0p5,METHOD=opd,OPD_LOSS_TYPE=weighted,OPD_TOKEN_FILTER=all,OPD_GEOMETRY_ALPHA=0.5"
