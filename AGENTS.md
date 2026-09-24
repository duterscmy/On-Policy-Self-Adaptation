# AGENTS.md

## Project Overview

This workspace contains the OPSA / OPD research environment on the Arrhenius EuroHPC cluster.

The workspace root is:

```text
/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa
```

The main development repository is:

```text
/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/slime-upstream
```

Most code changes should be made in `slime-upstream`.

The current research goal is to study token-level learning signals in on-policy distillation and OPSA-style training, especially whether low-log-probability tokens are intrinsically more useful or are favored by the geometry of the standard sampled reverse-KL objective.

---

## Workspace Layout

```text
opsa/
├── slime-upstream/       # MAIN development repo; current upstream Slime + OPSA port
├── slime/                # legacy/custom OPSA implementation; reference only unless asked
├── sglang/               # patched SGLang source used by current runtime
├── Megatron-LM/          # patched Megatron source used by current runtime
├── TransformerEngine/    # TransformerEngine source/build reference
├── sgl-router/           # ARM-built sglang router source
├── models/               # model checkpoints; DO NOT commit
├── data/                 # datasets; DO NOT commit
└── logs/                 # training/smoke logs; normally DO NOT commit
```

Important:

- Treat `slime-upstream` as the primary Git repository.
- Do NOT turn the whole `/opsa` workspace into one Git repository.
- Do NOT casually reset, checkout, or clean `sglang` or `Megatron-LM`; both contain required Slime patches.
- Do NOT modify or delete `models/`, `data/`, or large checkpoint files unless explicitly requested.

---

## Main Git Repository

Before editing code:

```bash
cd /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/slime-upstream
git status
git branch --show-current
```

Current OPSA migration work is expected to live on a branch such as:

```text
opsa-port
```

Use small, reviewable commits.

Never run destructive commands such as:

```bash
git reset --hard
git clean -fdx
```

unless explicitly requested.

Do not overwrite unrelated user changes.

---

## Runtime Environment

Conda environment:

```text
opsa-slime
```

Activate with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opsa-slime
```

Important runtime paths:

```bash
export OPSA_ROOT=/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa
export SLIME_DIR="$OPSA_ROOT/slime-upstream"
export SGLANG_DIR="$OPSA_ROOT/sglang"
export MEGATRON_DIR="$OPSA_ROOT/Megatron-LM"

export CUDA_HOME=/software/sse2/el9_gh200/easybuild/prefix/software/CUDA/12.9.1
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

export CUDNN_ROOT="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn"
export NCCL_HOME=/software/sse2/el9_gh200/easybuild/prefix/software/NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1
export GCC14_LIB=/software/sse2/el9_gh200/easybuild/prefix/software/GCCcore/14.3.0/lib64
export TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"

export LD_LIBRARY_PATH="$GCC14_LIB:$CUDNN_ROOT/lib:$NCCL_HOME/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SGLANG_DIR/python:${PYTHONPATH:-}"
```

Do NOT load the NVHPC build environment for normal training jobs.

The build module may be needed only for compilation:

```text
GPU/buildenv-nvhpc/25.9-cu12.9.1-eb
```

---

## Verified Core Versions

The working environment has been validated with approximately:

```text
Python                  3.12
PyTorch                 2.11.0+cu129
CUDA                    12.9
cuDNN                   9.17.1.4
SGLang                  0.5.15.post1
sglang-kernel           0.4.4+cu129
Transformer Engine      2.16.1+c9877be
sglang_router           0.3.2+slime
```

Important:

- Do NOT downgrade cuDNN to `9.16.x`.
- PyTorch was compiled against cuDNN 9.17.1 and failed with cuDNN 9.16.
- Apex was built successfully with GCC 13.
- Runtime uses GCC 14 `libstdc++` to satisfy Transformer Engine.
- Avoid replacing working compiled packages unless there is a concrete need.

---

## Patched Dependency State

### SGLang

Current SGLang source:

```text
/opsa/sglang
```

Current tested source commit:

```text
0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
```

For the current Slime/SGLang stack, these patches were applied from:

```text
slime-upstream/docker/patch/v0.5.15.post1/
```

Applied SGLang patches:

```text
sglang.patch
sglang-top_p.patch
sglang-release_hicache.patch
sglang-pull_weights.patch
sglang-deterministic.patch
```

Do NOT apply the legacy `v0.5.9` patches.

Do NOT blindly use `docker/patch/latest`; some patches there were already found incompatible with this stack.

### Megatron-LM

Current source:

```text
/opsa/Megatron-LM
```

Current tested commit:

```text
1dcf0dafa884ad52ffb243625717a3471643e087
```

Applied patches:

```text
megatron.patch
megatron-sglang-aligned.patch
```

Do not reset this tree without preserving/reapplying the patches.

---

## Model and Data

Qwen3-1.7B Hugging Face checkpoint:

```text
/opsa/models/Qwen3-1.7B
```

Converted Megatron distributed checkpoint:

```text
/opsa/models/Qwen3-1.7B_torch_dist
```

DAPO math data:

```text
/opsa/data/dapo-math-17k
```

Small smoke data may exist as:

```text
/opsa/data/dapo-math-17k/smoke-8.jsonl
```

Model architecture config:

```text
slime-upstream/scripts/models/qwen3-1.7B.sh
```

Qwen3-1.7B uses tied embeddings/output weights. Do not add:

```text
--untie-embeddings-and-output-weights
```

unless there is a specific reason.

---

## Cluster Usage Rules

### Login Node

Use the login node for:

- reading/editing code
- Git operations
- lightweight Python checks
- `compileall`
- CPU unit tests
- inspecting logs
- preparing scripts

Do NOT run GPU training on the login node.

### GPU Compute Node

Use a GPU node for:

- SGLang rollout
- Megatron forward/backward
- smoke tests
- training
- CUDA-dependent integration tests

The current hardware is GH200/aarch64.

A 2-GPU node is sufficient for the current Qwen3-1.7B smoke test:

```text
1 GPU actor
1 GPU rollout
TP = 1
```

Use larger allocations only when needed.

---

## SGLang Attention Backend on GH200

The current aarch64 `sgl_kernel` package does not expose the FA3 `flash_ops` module expected by SGLang.

This failure was observed:

```text
ImportError: cannot import name 'flash_ops' from 'sgl_kernel'
ImportError: Can not import FA3 in sgl_kernel
```

For current smoke/training runs, force rollout SGLang to:

```text
--sglang-attention-backend triton
```

Do not remove this workaround unless FA3 has been intentionally installed and verified.

Megatron's own attention backend is separate and may still use:

```text
--attention-backend flash
```

---

## Current OPSA Migration Status

The original baseline OPSA implementation has been ported into the current `slime-upstream`.

The first migration intentionally includes only:

```text
baseline OPSA
  - entropy mode
  - fixed-advantage mode
  - lowest current-actor-logprob token selection
```

It intentionally does NOT yet include:

```text
top-k OPSA experiments
sequence-level top-k experiments
GC-OPD
Bernoulli OPD
high-confidence-only OPD experiments
```

Relevant current files include:

```text
slime/utils/arguments.py
slime/backends/megatron_utils/actor.py
slime/backends/megatron_utils/model.py
slime/backends/megatron_utils/loss.py
slime/backends/megatron_utils/opsa.py
slime/rollout/opsa.py
tests/test_opsa_baseline.py
```

The baseline CPU unit tests pass.

A fixed-mode end-to-end OPSA smoke test has also passed successfully.

Observed correct fixed-mode smoke metrics:

```text
opsa/selected_fraction     0.19921875
opsa/selected_tokens       204
opsa/valid_tokens          1024
opsa/advantage_mean        -0.5
train/opsa/negative_fraction 1.0
train/opsa/positive_fraction 0.0
```

The run completed backward, optimizer step, weight synchronization, and Ray job success.

---

## Important OPSA Implementation Detail

`opsa_loss_mask` must remain floating point.

A previous bug converted it to the integer dtype of the original response mask and caused:

```text
RuntimeError: mean(): could not infer output dtype.
Input dtype must be either a floating point or complex dtype. Got: Int
```

The correct implementation keeps the generated OPSA mask as float, e.g.:

```python
opsa_masks = [
    x.reshape_as(loss_mask)
    for x, loss_mask in zip(flat_opsa_mask.split(split_sizes), loss_masks, strict=True)
]
```

Do not reintroduce `.to(dtype=loss_mask.dtype)` here.

---

## OPSA Baseline Semantics

Baseline OPSA operates on current actor log probabilities.

For each DP-local packed response batch:

1. Recompute current actor sampled-token log probabilities.
2. Restrict to valid response tokens.
3. Rank tokens by current actor log probability.
4. Select the lowest configured fraction, normally:

```text
bottom 20%
```

5. Assign either:
   - a fixed negative advantage, or
   - entropy-ranked negative advantages.
6. Normalize the policy-gradient loss over selected OPSA tokens, not over the full response.

Typical fixed smoke configuration:

```text
--advantage-estimator opsa
--opsa-mode fixed
--opsa-token-fraction 0.2
--opsa-fixed-advantage -0.5
--kl-coef 0
--entropy-coef 0
```

OPSA is reference-free in this configuration.

---

## Current Smoke Test Script

Current baseline smoke script:

```text
slime-upstream/examples/opsa/run-qwen3-1.7B-opsa-smoke.sh
```

Default expected allocation:

```text
ACTOR_GPUS=1
ROLLOUT_GPUS=1
TP_SIZE=1
```

Run from a GPU compute node:

```bash
cd /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/slime-upstream
bash examples/opsa/run-qwen3-1.7B-opsa-smoke.sh
```

Entropy mode can be tested with:

```bash
OPSA_MODE=entropy \
bash examples/opsa/run-qwen3-1.7B-opsa-smoke.sh
```

Before changing OPSA logic, preserve the existing fixed-mode smoke behavior.

---

## Current Research Direction

The main research question is whether the apparent importance of low-log-probability tokens in sampled reverse-KL OPD is partly caused by optimization geometry.

For standard detached sampled reverse-KL OPD:

```math
A_t = \log p_T(y_t) - \log p_S(y_t)
```

with surrogate:

```math
L_t = -\operatorname{sg}(A_t)\log p_S(y_t)
```

the sampled-token logit gradient contains:

```math
A_t(1-p_S(y_t))
```

Therefore high-confidence student tokens may be strongly suppressed by the softmax factor `1 - p_S`.

Planned experimental variants, after baseline OPSA is stable:

### 1. Vanilla OPD

```math
g \propto (\log p_T-\log p_S)(1-p_S)
```

### 2. Geometry-corrected OPD

Use a log-odds surrogate:

```math
L = -\operatorname{sg}(A)\log\frac{p}{1-p}
```

giving sampled-token gradient:

```math
g \propto A
```

This removes the explicit `1-p` suppression.

### 3. High-confidence-only OPD

Natural first threshold:

```text
p_S > 0.5
```

Test both:

```text
high-confidence + vanilla geometry
high-confidence + geometry-corrected geometry
```

Do not separate by advantage sign unless explicitly requested.

### 4. Bernoulli OPD

Use:

```math
L_{\mathrm{Bern}} =
-p_T\log p_S-(1-p_T)\log(1-p_S)
```

with sampled-token gradient:

```math
p_S-p_T
```

This replaces the log-ratio signal with ordinary probability gap.

---

## Preferred Experimental API

Do not create separate duplicated trainers for each method.

Prefer one parameterized implementation such as:

```text
--opd-loss-type vanilla
--opd-loss-type geometry_corrected
--opd-loss-type bernoulli

--opd-token-filter all
--opd-token-filter high_conf
--opd-token-filter bottom_percent
```

Keep the implementation modular and easy to ablate.

Add diagnostic logging for:

```text
student probability pS
teacher probability pT
log-ratio advantage A
pT - pS
selected-token fraction
probability bins
advantage statistics
gradient proxies / actual gradient checks where practical
```

Do not force a Top-K framing into experiments unless specifically requested.

---

## Development Workflow

For any nontrivial code change:

1. Inspect the existing implementation first.
2. Make the smallest coherent change.
3. Run lightweight checks before GPU tests.
4. Use CPU tests whenever possible.
5. Run a small GPU smoke test before larger jobs.
6. Inspect metrics, not only exit status.
7. Commit only after the behavior is verified.

Recommended lightweight checks:

```bash
python -m compileall -q slime
pytest -q tests/test_opsa_baseline.py
git diff --check
git status --short
```

For new loss functions, add CPU-level mathematical/unit tests before running GPU jobs.

---

## Safety Around Existing Code

Before modifying `loss.py`, `actor.py`, `model.py`, SGLang, or Megatron:

- inspect current diff
- understand existing patches
- avoid broad rewrites
- preserve upstream behavior for non-OPSA/non-OPD paths
- keep changes behind explicit feature flags where possible

Do not silently change experiment semantics.

If an implementation choice is ambiguous, explain the choice before making a large or hard-to-reverse change.

---

## Legacy OPSA Reference

The older custom implementation is under:

```text
/opsa/slime
```

It may contain historical experiments including Top-K and sequence-level variants.

Use it as a reference when needed, but do not assume its APIs can be copied directly into current Slime.

Known historical patch files include:

```text
/opsa/slime/opsa_topk_k10_4gpu.patch
/opsa/slime/seq_topk_sequence_level.patch
```

These include later experimental changes and should not be treated as the clean baseline OPSA definition.

---

## Logs and Debugging

Training/smoke logs are under:

```text
/opsa/logs
```

When debugging:

- find the first real traceback, not only the final Ray wrapper error
- distinguish warnings from fatal errors
- preserve successful smoke logs when they establish a known-good baseline

A successful OPSA smoke should show:

```text
rollout success
current actor logprob recomputation
OPSA selection metrics
finite loss and grad norm
backward / optimizer step
update_weights
SGLang update_weights HTTP 200
Ray job succeeded
```

---

## Git / GitHub Policy

GitHub should contain:

- source code
- experiment scripts
- tests
- small configuration files
- patch files
- reproducibility notes
- environment/version snapshots when useful

GitHub should NOT contain:

- model checkpoints
- datasets
- Conda environments
- compiled binary artifacts
- large logs
- Ray temporary files
- Hugging Face caches
- build directories

The server is the authoritative execution environment.

Git/GitHub is the authoritative version-control history.

A local laptop clone may be used for browsing and editing, but it is not expected to reproduce the GH200 runtime.

---

## Default Agent Behavior

When working in this project:

- Prefer editing `slime-upstream`.
- Read before writing.
- Preserve working environment assumptions.
- Do not reinstall or upgrade packages unless necessary.
- Do not reset patched dependency repositories.
- Do not run GPU workloads on login nodes.
- Do not create large files in Git.
- Do not duplicate implementations unnecessarily.
- Add tests for new mathematical objectives.
- Use small smoke tests before full experiments.
- Report the exact files changed and validation performed.
- If a GPU allocation is needed and none is available, prepare the code/tests first and clearly state what requires a GPU node.
