#!/usr/bin/env bash
set -euo pipefail

# Submit the controlled comparison serially so the suite uses at most one
# four-GPU node at a time. A failed condition stops the dependency chain.

OPSA_ROOT="${OPSA_ROOT:-/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa}"
SLIME_DIR="${SLIME_DIR:-$OPSA_ROOT/slime-upstream}"
RUNNER="$SLIME_DIR/examples/opsa/run-qwen3-1.7B-opd-opsa-study.sh"
ACCOUNT="${SLURM_ACCOUNT:-ehpc-reg-2026r01-278-gpu}"
PARTITION="${SLURM_PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

[[ -x "$RUNNER" ]] || { echo "ERROR: runner is not executable: $RUNNER" >&2; exit 1; }
mkdir -p "$OPSA_ROOT/logs"

previous_job=""

submit_condition() {
  local name="$1"
  local exports="$2"
  local dependency_args=()
  if [[ -n "$previous_job" ]]; then
    dependency_args=(--dependency="afterok:$previous_job")
  fi

  previous_job="$(
    sbatch --parsable \
      -A "$ACCOUNT" \
      -p "$PARTITION" \
      --nodes=1 \
      --gres=gpu:4 \
      --time="$TIME_LIMIT" \
      --job-name="$name" \
      --output="$OPSA_ROOT/logs/study-%x-%j.log" \
      --export="ALL,$exports" \
      "${dependency_args[@]}" \
      "$RUNNER"
  )"
  echo "$name $previous_job"
}

# Baselines first, then the five-condition OPD matrix, then the optional
# alpha=0.5 interpolation requested for the weighting ablation.
submit_condition "opd-vanilla" \
  "RUN_NAME=opd-vanilla-all,METHOD=opd,OPD_LOSS_TYPE=vanilla,OPD_TOKEN_FILTER=all"
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

echo "last dependency-chain job: $previous_job"
