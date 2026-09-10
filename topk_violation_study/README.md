# Top-K Violation Study

## Research question

**Does sampling a token outside the model's own Top-K candidate set predict eventual rollout failure?**

For generated token \(y_t\), define

\[
Z_t^{(K)}=\mathbf 1[\operatorname{rank}_{p_\theta(\cdot\mid x,y_{<t})}(y_t)>K].
\]

The final outcome is whether the complete response gives the correct answer.

The script reports both the naive accuracy gap and a **within-question paired effect** across repeated rollouts of the same prompt:

\[
\Delta_K=\mathbb E_q[\operatorname{Acc}(Z^{(K)}=1\mid q)-\operatorname{Acc}(Z^{(K)}=0\mid q)].
\]

This matters because hard questions may naturally produce both more tail events and lower accuracy. The within-question statistic removes much of that question-difficulty confound.

It also tests **early-prefix violations** in the first 32/64/128/256 generated tokens against final correctness, which gives a cleaner temporal story than looking only at whole-trajectory violation counts.

## Critical sampling detail

OPSA evaluates Qwen3 in **non-thinking mode** with 32 responses per prompt, temperature 0.7, top-k 20, top-p 0.8, and max response length 32,768.

That exact decoding distribution is **not suitable for the core tail analysis** because top-k=20 makes rank>20 impossible and top-p=0.8 further censors the tail. Therefore:

- `tail_analysis` (recommended for the scientific question): temperature 0.7, top-k=-1, top-p=1.0.
- `opsa` (sanity check only): temperature 0.7, top-k=20, top-p=0.8.

The prompt handling remains Qwen3 non-thinking mode (`enable_thinking=False`).

## Datasets and repeats

- **AIME24:** 30 questions × 32 samples = 960 trajectories/model. This matches OPSA's repeat count and is useful for Avg@32-style analysis, but only 30 distinct questions means larger uncertainty for question-level statistics.
- **MATH-500:** 500 questions × 8 samples = 4,000 trajectories/model. This is the strongest default dataset for the within-question predictive analysis.
- **GPQA Diamond:** supported with 8 samples/question. It is gated on Hugging Face; accept its terms and authenticate first. The prompt follows lm-eval-harness's generative CoT-zero-shot GPQA formatting, while Qwen3 is still forced into non-thinking mode.

## Install (Python 3.11)

```bash
conda create -n topk-study python=3.11 pip -y
conda activate topk-study
pip install -r requirements_topk_violation.txt
```

## First run a smoke test

```bash
bash smoke_test.sh
```

## Full default study

```bash
bash run_study.sh
```

This runs:

- Qwen/Qwen3-1.7B
- Qwen/Qwen3-4B
- AIME24 with 32 samples/prompt
- MATH-500 with 8 samples/prompt
- K = 1, 2, 5, 10, 20, 50, 100

Useful overrides:

```bash
TP_SIZE=2 MAX_TOKENS=8192 bash run_study.sh
```

For a cheap first pass, `MAX_TOKENS=8192` is sensible. For the final OPSA-aligned response cap, use 32768.

## OPSA-style sanity check

```bash
bash run_opsa_sanity.sh
```

This uses the exact OPSA-style sampling parameters on AIME24. Its purpose is to check that your local prompt/scoring setup gives base-model performance in the expected ballpark. Do **not** use this run to claim anything about K>=20 tail violations.

## GPQA Diamond

After accepting the dataset terms:

```bash
hf auth login

python run_topk_violation.py \
  --model Qwen/Qwen3-4B \
  --dataset gpqa_diamond \
  --samples-per-prompt 8 \
  --sampling-preset tail_analysis \
  --output-dir outputs/topk_violation/Qwen3-4B/gpqa_diamond
```

## Main outputs

Each run writes:

- `raw_rollouts.jsonl.gz`: generated responses plus the exact vLLM rank of every sampled token.
- `trajectory_metrics.csv`: per-response × K violation statistics.
- `summary.csv`: whole-trajectory predictive statistics.
- `prefix_summary.csv`: early-prefix violation → final outcome statistics.
- `count_summary.csv`: accuracy by number of violations for the primary K.
- `first_position_summary.csv`: accuracy by first-violation position.
- `figures/*.png`: standalone plots ready for PPT/paper use.

The most useful `summary.csv` columns are:

- `fraction_with_any_violation`
- `acc_group0`: accuracy for rollouts with no violation
- `acc_group1`: accuracy for rollouts with >=1 violation
- `within_problem_acc_delta`: question-controlled accuracy difference (violation minus no violation)
- `within_problem_ci_low/high`: 95% problem-bootstrap interval
- `failure_risk_ratio_group1_over_group0`

A negative `within_problem_acc_delta` means that, **for the same benchmark questions**, trajectories containing a Top-K violation tend to be less accurate.

## Figures

The run automatically creates separate figures (no multi-panel layout):

- `...__accuracy_by_violation.png`
- `...__within_problem_effect.png`
- `...__prefix_predictiveness_K10.png`
- `...__count_K10.png`
- `...__first_position_K10.png`

After multiple model/dataset runs, aggregate them with:

```bash
python aggregate_topk_violation.py \
  --root outputs/topk_violation \
  --primary-k 10
```

## What would count as a strong motivating result?

The cleanest story would be:

1. moderate-K violations (e.g. K=5–20) have a consistently negative within-question accuracy effect;
2. violations occurring **early** in the response also predict lower final accuracy;
3. the result replicates across Qwen3-1.7B and Qwen3-4B and across AIME24/MATH-500.

That would motivate the method as **on-policy tail-risk control / plausible-support regularization**, rather than merely "changing OPSA's selector from bottom-20% to Top-K rank." It is still correlational evidence; the causal test is the later training ablation comparing rank-based suppression with matched-rate low-logp suppression.
