# Project Handoff: Arrhenius OPSA / OPD

Last updated: 2026-09-25

This file supplements `AGENTS.md` and `RESEARCH_SPEC.md`. It records the
strict runtime boundary, the current known-good baselines, and the immediate
implementation sequence. If an older operational note conflicts with this
handoff, use the stricter rule here.

## 1. Architecture and runtime boundary

The Arrhenius login node and GH200 compute nodes are not the same
architecture/runtime environment. The working `opsa-slime` environment was
built for the GH200/aarch64 compute environment.

Use the login node only for:

- Git operations;
- reading and editing text/source files;
- `grep`, `rg`, `find`, `sed`, and `awk`;
- inspecting logs and diffs;
- preparing scripts and configuration;
- submitting and querying Slurm jobs;
- shell-only checks that do not execute the aarch64 Python environment.

Do not use the login node to validate the research runtime. In particular, do
not run these there:

```text
conda activate opsa-slime
python -c "import torch"
python -c "import sglang"
python -c "import transformer_engine"
pytest ...
pip install ...
python setup.py ...
```

Anything that imports, tests, installs, or builds the GH200/aarch64 stack must
run through Slurm on an appropriate compute node. A login-node failure is not
evidence that the compute-node environment is broken.

Compute-node-only work includes:

- activating `opsa-slime`;
- PyTorch/CUDA, SGLang, Megatron-LM, TransformerEngine, and Apex imports;
- tests that import the research stack;
- compiled extensions and package installation;
- model conversion;
- rollout, training, and GPU smoke tests.

Do not load the NVHPC build module for ordinary runtime or training jobs.

## 2. Slurm usage

Codex may request compute resources itself. Inspect the existing allocation
scripts and cluster state before submitting; do not invent account or
partition values, and do not depend on interactive shell aliases.

Current explicit allocation parameters are:

```text
account:   ehpc-reg-2026r01-278-gpu
partition: gpu
hardware:  NVIDIA GH200 120GB
```

Useful read-only checks:

```bash
sinfo
squeue -u "$USER"
```

For iterative debugging, prefer one allocation followed by repeated
`srun --jobid=<JOBID> ...` commands, then release it. For one-shot tests,
`srun` or `sbatch` is easier to automate.

For current Qwen3-1.7B rollout/training smoke tests, two GPUs are normally
enough:

```text
1 GPU actor
1 GPU rollout
TP = 1
```

Request more only when the experiment requires it.

## 3. Known-good baseline state

### Vanilla OPD

A full Qwen3-1.7B Vanilla OPD smoke test has completed successfully, including:

- rollout;
- Megatron actor execution;
- teacher/student OPD path;
- training step and optimizer update;
- `update_weights`;
- SGLang weight synchronization;
- successful Ray job completion.

Known-good log:

```text
/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/logs/
qwen3-1.7b-opd-smoke-20260924-131612.log
```

### Fixed-mode OPSA

A full fixed-mode baseline OPSA smoke test has completed successfully.

Observed metrics:

```text
rollout/opsa_loss_mask               0.19921875
rollout/opsa/selected_fraction       0.19921875
rollout/opsa/selected_tokens         204
rollout/opsa/valid_tokens            1024
rollout/opsa/advantage_mean          -0.5

train/opsa/selected_fraction         0.19921875
train/opsa/advantage_mean            -0.5
train/opsa/negative_fraction         1.0
train/opsa/positive_fraction         0.0
```

The run had finite loss and gradient norm, completed an optimizer step,
synchronized weights with HTTP 200 responses, and ended with Ray success.

Known-good log:

```text
/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/logs/
qwen3-1.7b-opsa-fixed-smoke-20260924-232016.log
```

Preserve this behavior as a regression baseline.

### Entropy-mode OPSA

Entropy-mode OPSA is not yet an established end-to-end baseline. Do not
describe it as verified until its smoke test passes and the metrics are
inspected.

## 4. Critical regression invariants

### OPSA mask dtype

`opsa_loss_mask` must remain floating point. Do not cast it to the original
integer response-mask dtype.

Correct:

```python
opsa_masks = [
    x.reshape_as(loss_mask)
    for x, loss_mask in zip(flat_opsa_mask.split(split_sizes), loss_masks, strict=True)
]
```

Do not reintroduce:

```python
.to(dtype=loss_mask.dtype)
```

The old cast caused:

```text
RuntimeError: mean(): could not infer output dtype.
Input dtype must be either a floating point or complex dtype. Got: Int
```

The regression is covered by `tests/test_opsa_baseline.py`.

### SGLang attention backend

The current aarch64 `sgl_kernel` package does not expose the FA3 `flash_ops`
module. Rollout must use:

```text
--sglang-attention-backend triton
```

Keep this until FA3 is intentionally rebuilt and verified. Megatron's separate
attention backend may still use `--attention-backend flash`.

## 5. Git and repository state

Primary development happens in:

```text
/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/slime-upstream
```

Before substantial changes, record:

```bash
git status --short
git branch --show-current
git log -1 --oneline
git diff
```

The GitHub repository is:

```text
git@github.com:duterscmy/On-Policy-Self-Adaptation.git
```

Use:

- `main` for the root OPSA project, documentation, and research context;
- `opsa-port` for the current upstream Slime implementation and experiments.

Do not upload models, datasets, checkpoints, environments, dependency source
trees, large logs, Ray state, or temporary archives. Preserve unrelated user
changes and never use destructive Git commands without explicit instruction.

## 6. OPD confidence-rebalancing workflow

Preserve default Vanilla OPD behavior exactly. Keep loss geometry independent
from token filtering:

```text
loss geometry:
    vanilla
    geometry_corrected
    bernoulli

token filter:
    all
    high_conf
    bottom_percent
```

Implement and validate incrementally:

```text
inspect current Vanilla OPD code
        ↓
map equations to exact implementation
        ↓
write Vanilla gradient regression test
        ↓
implement geometry-corrected objective
        ↓
CPU gradient test on a compute node
        ↓
Vanilla smoke regression
        ↓
GC smoke
        ↓
high-confidence filter
        ↓
high-conf Vanilla vs high-conf GC
        ↓
Bernoulli objective
        ↓
probability-bin diagnostics
        ↓
larger experiments
```

Add one mechanism at a time. Do not mix all variants into one large commit or
duplicate trainers. Keep optimizer, learning rate, rollout settings, data,
seeds, and evaluation fixed across comparisons unless a difference is
explicitly justified.

Log gradient norm and loss scale because geometry correction changes effective
gradient magnitude. Do not tune one method more heavily without documenting
the difference.

## 7. Experiment records

Every GPU experiment should record at least:

```text
git commit and dirty diff state
model checkpoint
teacher checkpoint
dataset
seed
rollout batch size
samples per prompt
global batch size
learning rate
number of steps / rollouts
loss type
token filter
threshold or fraction
GPU count
TP size
exact command or script
log path
```

Do not rely only on shell history.

After coding/debugging, report:

- exact files changed;
- behavior implemented;
- exact checks/tests and their outcomes;
- whether validation ran on the login node or a GH200 compute node;
- relevant smoke metrics, loss, gradient norm, weight synchronization, and
  Ray status;
- branch, commit, and remaining uncommitted changes;
- anything not yet verified.

## 8. Environment stability

Do not broadly upgrade or rebuild PyTorch, CUDA, cuDNN, SGLang, Megatron-LM,
TransformerEngine, or Apex while modifying a research loss unless there is
concrete evidence and a documented reason. The current environment is known to
work. Treat new training failures as code-path or configuration issues first.

## 9. Scientific discipline

The goal is to test the hypothesis, not to make geometry correction look
better. Keep negative results, avoid overstating correlation evidence, and
preserve the distinction between teacher disagreement and optimization
strength.

The core question remains:

> Does OPD focus on low-confidence tokens because those tokens intrinsically
> contain more useful learning signal, or because the standard sampled
> reverse-KL objective assigns them disproportionately large gradients?

## 10. Current implementation and experiment queue (2026-09-25)

The parameterized OPD objective implementation is on `opsa-port` at commit:

```text
d3d1d670 Gate OPD study suite on smoke validation
c2788efb Add parameterized OPD objective study
```

Implemented loss types and filters:

```text
--opd-loss-type vanilla|geometry_corrected|bernoulli|weighted
--opd-token-filter all|high_conf|bottom_percent
--opd-high-conf-threshold 0.5
--opd-bottom-fraction 0.2
--opd-geometry-alpha 0.5
```

The default `vanilla + all` path remains on the legacy advantage/PPO path.
Non-default variants separate the OPD term from the base reward advantage so
OPD token filtering does not filter task-reward learning. The implementation
also logs student/teacher probabilities, log-ratio and probability-gap
statistics, ten student-probability bins, selected fractions, and analytic
gradient-mass proxies.

Validation completed on Slurm GH200 compute nodes:

```text
unit/gradient tests: job 2956338, 10 passed
four-GPU GC + high-confidence smoke: job 2956607, Ray succeeded
selected tokens in smoke: 917 / 1024 = 0.8955078125
```

The pinned AIME24 evaluation file is now present at:

```text
/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/opsa/data/aime-2024/aime-2024.jsonl
```

It has 30 rows and the required `prompt` and `label` fields.

The production launcher and serial submitter are:

```text
slime-upstream/examples/opsa/run-qwen3-1.7B-opd-opsa-study.sh
slime-upstream/examples/opsa/submit-qwen3-1.7B-opd-opsa-study.sh
```

The exact production-config smoke, job `2957249`, completed successfully and
released the first long run. The full suite is an `afterok` dependency chain,
one four-GPU node at a time, 450 rollouts and at most 24 hours per condition:

```text
2957349  Vanilla OPD, all tokens
2957350  fixed OPSA, bottom 20%
2957351  geometry-corrected OPD, all tokens
2957352  high-confidence Vanilla OPD
2957353  high-confidence geometry-corrected OPD
2957354  Bernoulli OPD, all tokens
2957355  weighted OPD, alpha=0.5, all tokens
```

The chain is intentionally fail-closed: a failed smoke or condition prevents
later jobs from starting and wasting GPU time. Slurm wrapper and inner Ray
logs are under `/opsa/logs`; checkpoints are under `/opsa/outputs` and are not
committed.
