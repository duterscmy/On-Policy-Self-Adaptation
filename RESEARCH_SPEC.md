# Research Specification: Confidence-Dependent Geometry in OPD and OPSA

## 1. Purpose

This document defines the research question, hypotheses, objectives,
diagnostics, implementation requirements, and experiment order for the next
stage of the OPSA / on-policy distillation (OPD) project.

Repository, cluster, and runtime rules are defined in `AGENTS.md` and the
Arrhenius / Slurm execution policy in `README.md`. The primary implementation
repository is `slime-upstream`.

## 2. Research question

OPD and OPSA training appear to be dominated by sampled tokens with low
student confidence (low current-actor log probability). OPSA exploits this
directly: it selects low-log-probability sampled tokens and applies negative
token-level learning signals only to those tokens. Training on a relatively
small bottom fraction can reproduce much of the benefit of full OPD.

This observation has two competing explanations:

1. **Intrinsic-signal hypothesis:** low-confidence tokens contain most of the
   useful learning information.
2. **Objective-geometry hypothesis:** the standard sampled reverse-KL
   surrogate systematically amplifies low-confidence tokens and suppresses
   high-confidence tokens, regardless of whether the latter contain useful
   teacher corrections.

The central question is:

> Does OPD focus on low-confidence tokens because those tokens intrinsically
> contain the useful learning signal, or because the standard sampled
> reverse-KL objective gives them disproportionately large gradients?

The project must keep **teacher disagreement (signal quality)** separate from
**gradient magnitude (optimization geometry)**.

## 3. Standard sampled OPD geometry

For the sampled token `y`, define

\[
p_S = p_S(y), \qquad p_T = p_T(y),
\]

and the detached OPD advantage

\[
A = \log p_T - \log p_S.
\]

The standard surrogate is

\[
L_{\mathrm{vanilla}} = -\operatorname{sg}(A)\log p_S.
\]

For the sampled-token logit `z_y`,

\[
\frac{\partial L_{\mathrm{vanilla}}}{\partial z_y}
= -A(1-p_S).
\]

The sampled-token gradient magnitude is therefore proportional to

\[
|A|(1-p_S)
= |\log p_T-\log p_S|(1-p_S).
\]

The full-vocabulary logit-gradient L1 norm is proportional to

\[
2|A|(1-p_S).
\]

Low-confidence tokens can consequently be favored twice:

- the log-ratio disagreement `|log p_T - log p_S|` is often larger in the
  low-probability region;
- the explicit softmax factor `1-p_S` suppresses high-confidence sampled
  tokens.

A high-confidence token can still satisfy `p_S >> p_T`, meaning that the
teacher wants its probability reduced, while receiving almost no standard
OPD gradient because `1-p_S` is close to zero.

## 4. Existing empirical evidence

Preliminary probability-bin analyses show:

- More than roughly 70% of sampled response tokens can fall in the
  `p_S in [0.9, 1.0]` bin.
- Mean absolute log-ratio disagreement decreases sharply with confidence;
  indicative observed values were about `1.6` in `[0, 0.1]` and `0.09` in
  `[0.9, 1.0]`.
- The signed probability gap `p_T-p_S` is often negative at high confidence,
  so the teacher still asks to decrease many highly confident student tokens.
- The standard proxy `|A|(1-p_S)` nearly collapses in the highest-confidence
  bins.
- Aggregate gradient-proxy mass is much more concentrated in low-confidence
  bins than token frequency alone would suggest.
- Bottom-token OPSA works surprisingly well, but this is consistent with both
  the intrinsic-signal and objective-geometry hypotheses.

These values are approximate observations from prior runs, not universal
constants. They must be regenerated and reported with the exact run and data
configuration.

Within-problem correlations between low-confidence/violation patterns and
correctness have been only moderate. They motivate causal training
experiments, but do not establish that only low-confidence tokens matter.

## 5. Falsifiable hypotheses

- **H1:** Standard OPD structurally underweights high-confidence tokens via
  `A(1-p_S)`.
- **H2:** High-confidence tokens retain non-trivial teacher correction signal,
  observable through quantities such as `p_T-p_S`.
- **H3:** Reducing or removing the `1-p_S` suppression makes useful
  high-confidence corrections more learnable.
- **H4:** If high-confidence training remains ineffective after geometry
  correction, the intrinsic-signal interpretation of OPSA becomes stronger.

No experiment should assume one interpretation is true in advance.

## 6. Objective family

### 6.1 Vanilla OPD

\[
A = \log p_T-\log p_S,
\qquad
L = -\operatorname{sg}(A)\log p_S,
\qquad
\frac{\partial L}{\partial z_y}=-A(1-p_S).
\]

This is the primary reference condition. The default code path must remain
identical to the current implementation.

### 6.2 Geometry-corrected OPD (GC-OPD)

Keep the detached log-ratio teacher signal, but use sampled-token log odds:

\[
L_{\mathrm{GC}}
= -\operatorname{sg}(A)\log\frac{p_S}{1-p_S}.
\]

Because

\[
\frac{\partial}{\partial z_y}
\log\frac{p_S}{1-p_S}=1,
\]

the sampled-token logit gradient is

\[
\frac{\partial L_{\mathrm{GC}}}{\partial z_y}=-A.
\]

GC-OPD preserves the original log-ratio disagreement while removing only the
explicit `1-p_S` suppression. It is the cleanest controlled test of the main
hypothesis.

### 6.3 Continuous geometry weighting

An optional continuous interpolation weights the vanilla surrogate with a
detached confidence weight

\[
w(p_S)=(1-p_S)^{-\alpha}.
\]

Its sampled-token gradient scales as

\[
A(1-p_S)^{1-\alpha}.
\]

Useful settings are:

- `alpha = 0`: Vanilla, `A(1-p_S)`;
- `alpha = 0.5`: partial correction, `A sqrt(1-p_S)`;
- `alpha = 1`: full sampled-logit correction, `A`.

The confidence-derived weight must be detached if this exact interpretation
is intended. Weight normalization should be considered so that the overall
loss scale remains comparable across conditions.

This sweep is preferable to arbitrary multipliers such as 5x, 10x, or 20x,
but it is an optional ablation after the main objectives are stable.

### 6.4 High-confidence-only training

Select tokens with

\[
p_S > 0.5.
\]

The threshold has a natural interpretation: the sampled token alone carries
more than half of the probability mass and must be top-1.

Two conditions are essential:

- **High-confidence Vanilla:** retain `-A log p_S` and its `1-p_S`
  suppression.
- **High-confidence GC:** use `-A log(p_S/(1-p_S))` and remove the
  suppression.

Their contrast is one of the most diagnostic experiments. Do not split by
the sign of `A` in the main experiment.

### 6.5 Bernoulli OPD

Treat the sampled token versus all other tokens as a Bernoulli problem:

\[
L_{\mathrm{Bern}}
= -p_T\log p_S-(1-p_T)\log(1-p_S),
\]

where `p_T` is detached. Its sampled-token logit gradient is

\[
\frac{\partial L_{\mathrm{Bern}}}{\partial z_y}=p_S-p_T.
\]

Bernoulli OPD removes both the explicit `1-p_S` suppression and the
low-probability amplification of the log-ratio signal. It is a diagnostic
control for separating the effect of softmax geometry from the effect of the
teacher signal parameterization.

The three central geometries are:

| Objective | Effective sampled-token signal |
|---|---|
| Vanilla OPD | `(log p_T - log p_S)(1-p_S)` |
| GC-OPD | `log p_T - log p_S` |
| Bernoulli OPD | `p_T-p_S` |

## 7. Baselines and minimum experiment matrix

Baseline reproduction should happen before new objectives:

1. Vanilla OPD baseline.
2. Existing OPSA baseline, with the exact fixed or entropy mode and token
   fraction reported; its semantics must not be changed.

The first clean objective suite is:

| Condition | Token set | Teacher signal | Sampled-token gradient |
|---|---|---|---|
| Vanilla OPD | all | log ratio | `-A(1-p_S)` |
| GC-OPD | all | log ratio | `-A` |
| High-conf Vanilla | `p_S > 0.5` | log ratio | `-A(1-p_S)` |
| High-conf GC | `p_S > 0.5` | log ratio | `-A` |
| Bernoulli OPD | all | probability target | `p_S-p_T` |

OPSA is an external low-confidence baseline alongside this objective suite.
The optional `alpha in {0, 0.5, 1}` weighting sweep should come only after
the five main OPD conditions are stable.

## 8. Required token-level diagnostics

Use student-confidence bins

\[
[0,0.1), [0.1,0.2), \ldots, [0.9,1.0].
\]

For each bin, report:

### Token distribution

- valid-token count;
- fraction of all valid response tokens.

### Teacher-student disagreement

- mean signed `A = log p_T-log p_S`;
- mean `|A|`;
- mean signed `p_T-p_S`;
- mean `|p_T-p_S|`.

### Geometry-specific optimization proxies

- mean and total `|A|(1-p_S)`;
- fraction of total Vanilla proxy mass;
- mean and total `|A|` for GC;
- mean and total `|p_T-p_S|` for Bernoulli.

### Training selection and scale

- selected-token fraction overall and by confidence bin;
- mean advantage or target signal by bin;
- loss scale;
- gradient norm;
- update norm where available;
- actual sampled-token gradient statistics where practical.

The existing Stage-1 / Stage-2 probing workflow should be extended to show
`|A|(1-p_S)`, `|A|`, and `|p_T-p_S|` side by side.

Downstream evaluation must include accuracy/pass@1, pass@k where appropriate,
training correctness/reward, and stability, but downstream scores alone are
not sufficient. Each result must be connected to where optimization mass was
allocated.

## 9. Controlled-comparison requirements

Keep the following fixed whenever possible:

- base model and teacher;
- training data and rollout distribution;
- rollout temperature and samples per prompt;
- global batch size;
- number of training tokens/steps;
- learning rate;
- evaluation pipeline;
- random seeds.

Because correction changes gradient scale, prefer a principled normalization
over arbitrary per-method learning-rate tuning. Any normalization or tuning
must be reported explicitly.

Initial development uses Qwen3-1.7B and DAPO-Math-17k, with reasoning
evaluation such as AIME/AIME24 and MATH500. The first goal is a small,
reproducible causal result rather than a broad benchmark sweep.

## 10. Implementation design

Use one parameterized OPD implementation rather than separate trainers.

Proposed CLI:

```text
--opd-loss-type vanilla
--opd-loss-type geometry_corrected
--opd-loss-type bernoulli
--opd-loss-type weighted

--opd-token-filter all
--opd-token-filter high_conf
--opd-token-filter bottom_percent

--opd-high-conf-threshold 0.5
--opd-bottom-fraction 0.2
--opd-geometry-alpha 0.5
```

Requirements:

- default behavior remains exactly Vanilla OPD;
- every non-default behavior is behind an explicit flag;
- filtering and confidence weighting use detached student probabilities;
- probabilities are clamped safely for log/log-odds operations;
- OPSA baseline semantics remain unchanged;
- legacy Top-K and sequence-level variants are not folded into this work;
- code comments and metrics distinguish teacher disagreement from gradient
  magnitude.

## 11. Mathematical and regression tests

Before GPU runs, add CPU tests that construct a small logit vector and verify
autograd against the expected sampled-token gradients:

- Vanilla: `-A(1-p_S)`;
- GC: `-A`;
- Bernoulli: `p_S-p_T`.

Also test:

- default Vanilla output and gradient remain unchanged;
- masking and selected-token normalization;
- high-confidence filtering at the threshold boundary;
- bottom-percent filtering if implemented;
- edge cases near `p_S=0` and `p_S=1`;
- numerical clamping without NaN or Inf;
- weighted-objective gradients for selected alpha values;
- integer response masks do not turn generated loss masks into integer
  tensors.

## 12. Implementation and experiment order

Keep commits small and separable.

1. Stabilize and record the current OPSA baseline fix and regression test.
2. Inspect the existing Vanilla OPD path end to end:
   student log probabilities, teacher log probabilities, detached advantage,
   token loss, masks, and reduction.
3. Add CPU gradient/regression tests for current Vanilla OPD.
4. Run a tiny Vanilla OPD smoke test on a Slurm compute node.
5. Run and record the OPSA baseline with its exact configuration.
6. Refactor only enough to introduce the loss-type switch.
7. Implement GC-OPD and verify its exact gradient on CPU.
8. Run a tiny GC smoke test on a Slurm compute node.
9. Implement high-confidence filtering and the Vanilla/GC pair.
10. Implement Bernoulli OPD and its gradient tests.
11. Add probability-bin and gradient-mass diagnostics.
12. Run the five-condition minimum matrix.
13. Only then consider the continuous `alpha` sweep and larger experiments.

Login-node and compute-node restrictions in `README.md` and `AGENTS.md` apply
throughout. GPU-runtime imports, GPU smoke tests, and training must be launched
through Slurm on a compute node.

## 13. Falsifiable interpretation matrix

| Outcome | Interpretation |
|---|---|
| GC-OPD > Vanilla | Explicit `1-p_S` suppression likely hurts standard OPD. |
| High-conf GC > High-conf Vanilla | High-confidence tokens contain useful corrections that Vanilla under-optimizes. |
| Both high-conf variants fail | High-confidence tokens may genuinely contain relatively little useful signal. |
| Bernoulli > GC | Log-ratio teacher geometry may itself contribute to the imbalance. |
| GC > Bernoulli | Log-ratio disagreement remains useful after softmax suppression is removed. |
| Corrected variants approximately equal Vanilla | Other training effects may dominate, or the confidence skew may not be harmful. |

All of these outcomes are scientifically useful.

The strongest evidence for the geometry-bias hypothesis would be the complete
chain:

1. Vanilla allocates little gradient to high-confidence tokens.
2. Probability-gap diagnostics show meaningful teacher disagreement there.
3. High-confidence Vanilla is weak.
4. High-confidence GC is substantially stronger.
5. Full GC matches or improves Vanilla while redistributing optimization mass.

A meaningful negative result is also possible: geometry correction may change
gradient allocation without improving downstream performance. That would
suggest that low-confidence concentration is real but potentially beneficial.

## 14. Claims and terminology

Preferred language:

- confidence-dependent optimization bias;
- confidence-dependent gradient imbalance;
- systematic suppression of high-confidence corrections;
- objective-induced preference for low-confidence tokens;
- confidence-skewed optimization geometry;
- signal quality versus optimization geometry.

Do not claim prematurely that:

- high-confidence tokens are always useful;
- low-confidence tokens are unimportant;
- OPSA is wrong;
- GC-OPD must outperform Vanilla;
- the standard KL objective is theoretically invalid.

The current claim is narrower: standard sampled OPD has confidence-dependent
gradient geometry, and controlled experiments are needed to determine whether
that geometry under-optimizes useful high-confidence teacher corrections.

## 15. Possible paper framing

Potential titles include:

- *Revisiting Low-Confidence Tokens in On-Policy Distillation: Signal or
  Optimization Geometry?*
- *Confidence-Skewed Gradients in On-Policy Distillation*
- *Disentangling Token Informativeness from Loss Geometry in On-Policy
  Distillation*
- *Rebalancing High-Confidence Corrections in On-Policy Distillation*

The intended contribution is to quantify confidence-dependent gradient
imbalance, separate token frequency/disagreement/optimization-mass
distributions, introduce controlled geometry-corrected objectives, and use
them to reinterpret why low-confidence token selection works in OPSA.
