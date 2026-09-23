# OPD Log-Probability Probe

Two-stage codebase for testing whether OPD's low-logp dominance reflects genuine teacher/student disagreement or amplification by the reverse-KL log-ratio geometry.

## Stage 1 — GPU collection, run once

Default: AIME 2024, Qwen3-1.7B student, Qwen3-4B teacher, 8 stochastic rollouts/problem.

```bash
python scripts/check_gpu_env.py
python -m pip install -r requirements-stage1.txt
PYTHONPATH=src python stage1_collect.py --config configs/rollout_aime.yaml
```

For MATH-500:

```bash
PYTHONPATH=src python stage1_collect.py --config configs/rollout_math500.yaml
```

Stage 1 saves one Parquet shard per rollout under `runs/.../tokens/`, so it is resumable. It also saves raw rollout JSON files. Re-running skips completed Parquet shards.

Per-token fields include:

- student/teacher probability and log-probability
- student/teacher entropy
- exact selected-token rank and top-1 probability
- signed/absolute probability gap: `p_T - p_S`
- signed/absolute log-ratio: `log p_T - log p_S`
- signed/absolute sampled-token gradient proxy: `(log p_T - log p_S)(1-p_S)`

Teacher scoring is teacher-forced on the exact student-generated token IDs. The script requires identical tokenizer vocabularies/token IDs, which is appropriate for Qwen3 student/teacher pairs and avoids tokenizer alignment as a confound.

Default rollout settings intentionally use the full student distribution: temperature 1.0, top-p 1.0, top-k disabled (`top_k: 0`).

## Stage 2 — CPU analysis, rerun freely

```bash
python -m pip install -r requirements-stage2.txt
PYTHONPATH=src python stage2_analyze.py --config configs/analysis.yaml
```

It writes:

- `token_level_all.parquet`
- `binned_statistics.csv`
- `01_student_logprob_distribution.pdf`
- `02_absolute_probability_gap.pdf`
- `03_logratio_gap.pdf`
- `04_gradient_signal.pdf`

The first plot shows where sampled student tokens lie in log-probability space. The next two condition teacher/student disagreement on student log-probability, using ordinary probability space and log-ratio space respectively. The fourth compares per-token gradient intensity with each bin's share of total gradient-proxy mass.

Use `statistics.central: median` for robust plots; switch to `mean` when you specifically want sensitivity to rare large disagreements. The CSV always contains both mean and median plus IQR columns.

## GH200 notes

GH200 systems commonly use an ARM64 Grace CPU. The risky piece is installing a matching PyTorch/CUDA build, so `requirements-stage1.txt` deliberately does not install PyTorch. Prefer the cluster's working PyTorch module or an NVIDIA PyTorch NGC container, then pip-install only the Hugging Face/data dependencies. Avoid building FlashAttention unless your cluster already provides a known-good GH200 build; this repo uses PyTorch SDPA.

The default model pair is small relative to GH200 memory. If desired, change the teacher to `Qwen/Qwen3-8B` in the YAML.

## Suggested first run

Start with 10 AIME problems x 8 rollouts and `rollout_batch_size: 2`. Once verified, scale to all 30 AIME problems x 16 rollouts. MATH-500 is useful when you want smoother distributions from more problems.

## Smoke test for Stage 2

No GPU is needed:

```bash
PYTHONPATH=src python scripts/smoke_test_stage2.py
PYTHONPATH=src python stage2_analyze.py --config configs/analysis_smoke.yaml
```

## SLURM

`slurm/stage1_gh200.sbatch` is intentionally generic because account/partition names are cluster-specific. Add your own `#SBATCH --account=...` and `#SBATCH --partition=...` lines before submitting.
