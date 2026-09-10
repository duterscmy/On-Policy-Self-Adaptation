#!/bin/bash
#SBATCH --job-name="topk_violation_study"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

set -eo pipefail

source ~/.bashrc
conda activate topk-vllm085
set -u

OUT_ROOT="${OUT_ROOT:-outputs/topk_violation}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
K_VALUES="${K_VALUES:-1,2,3,5,10,20,50}"
PRIMARY_K="${PRIMARY_K:-10}"
PREFIX_TOKENS="${PREFIX_TOKENS:-32,64,128,256}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-2000}"

MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
)

run_one () {
  local model="$1"
  local dataset="$2"
  local n="$3"
  local short_model
  short_model="$(basename "${model}")"
  local out="${OUT_ROOT}/${short_model}/${dataset}"

  python run_topk_violation.py \
    --model "${model}" \
    --dataset "${dataset}" \
    --samples-per-prompt "${n}" \
    --sampling-preset tail_analysis \
    --max-tokens "${MAX_TOKENS}" \
    --k-values "${K_VALUES}" \
    --primary-k "${PRIMARY_K}" \
    --prefix-tokens "${PREFIX_TOKENS}" \
    --bootstrap-reps "${BOOTSTRAP_REPS}" \
    --output-dir "${out}" \
    --resume
}

mkdir -p "${OUT_ROOT}"

for model in "${MODELS[@]}"; do
  run_one "${model}" aime24 32
  run_one "${model}" math500 8

  # GPQA Diamond is gated on Hugging Face. After accepting its terms and
  # running `hf auth login`, uncomment the next line:
  # run_one "${model}" gpqa_diamond 8
done

python aggregate_topk_violation.py --root "${OUT_ROOT}" --primary-k "${PRIMARY_K}"
echo "Done: ${OUT_ROOT}"
