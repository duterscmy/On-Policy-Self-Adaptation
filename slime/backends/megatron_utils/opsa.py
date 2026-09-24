"""Baseline OPSA token selection and advantages.

OPSA operates on the response tokens in one data-parallel worker's local
packed batch. It selects the lowest current-actor sampled-token log-probabilities
across that batch and assigns advantages only to those selected tokens.
"""

import math
from dataclasses import dataclass

import torch


def validate_opsa_args(args) -> None:
    """Validate baseline OPSA CLI values and derive entropy requirements."""
    if args.advantage_estimator != "opsa":
        return

    if args.use_opd:
        raise ValueError("--advantage-estimator=opsa and --use-opd are mutually exclusive.")
    if args.normalize_advantages:
        raise ValueError("--advantage-estimator=opsa and --normalize-advantages are mutually exclusive.")
    if args.use_rollout_logprobs:
        raise ValueError(
            "OPSA requires log-probs recomputed by the current Megatron actor; "
            "disable --use-rollout-logprobs."
        )
    if args.kl_coef != 0 or args.use_kl_loss:
        raise ValueError("OPSA is reference-free; --kl-coef must be 0 and --use-kl-loss must be disabled.")
    if not 0 < args.opsa_token_fraction <= 1:
        raise ValueError("--opsa-token-fraction must be in (0, 1].")

    if args.opsa_mode == "entropy":
        if not args.opsa_advantage_min < args.opsa_advantage_max < 0:
            raise ValueError("--opsa-advantage-min/max must satisfy min < max < 0.")
        if args.opsa_fixed_advantage is not None:
            raise ValueError("--opsa-fixed-advantage is only valid with --opsa-mode=fixed.")
        args.use_rollout_entropy = True
    elif args.opsa_mode == "fixed":
        if args.entropy_coef != 0:
            raise ValueError("--opsa-mode=fixed requires --entropy-coef=0 so entropy is not computed.")
        if args.opsa_fixed_advantage is None or args.opsa_fixed_advantage == 0:
            raise ValueError("--opsa-mode=fixed requires a non-zero --opsa-fixed-advantage.")
        if args.use_rollout_entropy:
            raise ValueError("--opsa-mode=fixed does not compute entropy; remove --use-rollout-entropy.")
        args.use_rollout_entropy = False
    else:
        raise ValueError(f"Unsupported baseline OPSA mode: {args.opsa_mode!r}.")


@dataclass(frozen=True)
class OPSAOutput:
    """Per-sample OPSA tensors plus batch-level diagnostic metrics."""

    advantages: list[torch.Tensor]
    loss_masks: list[torch.Tensor]
    metrics: dict[str, torch.Tensor]


def _validate_inputs(
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    entropies: list[torch.Tensor] | None,
) -> None:
    if len(log_probs) != len(loss_masks):
        raise ValueError(
            f"OPSA expected one loss mask per log-prob tensor, got "
            f"{len(loss_masks)} masks for {len(log_probs)} log-prob tensors."
        )
    if entropies is not None and len(entropies) != len(log_probs):
        raise ValueError(
            f"OPSA expected one entropy tensor per log-prob tensor, got "
            f"{len(entropies)} entropy tensors for {len(log_probs)} log-prob tensors."
        )

    for index, (log_prob, loss_mask) in enumerate(zip(log_probs, loss_masks, strict=True)):
        if log_prob.shape != loss_mask.shape:
            raise ValueError(
                f"OPSA log-prob/loss-mask shape mismatch for sample {index}: "
                f"{tuple(log_prob.shape)} vs {tuple(loss_mask.shape)}."
            )
        if entropies is not None and entropies[index].shape != log_prob.shape:
            raise ValueError(
                f"OPSA entropy/log-prob shape mismatch for sample {index}: "
                f"{tuple(entropies[index].shape)} vs {tuple(log_prob.shape)}."
            )


def compute_opsa(
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    *,
    token_fraction: float,
    mode: str,
    entropies: list[torch.Tensor] | None = None,
    advantage_min: float = -1.0,
    advantage_max: float = -0.5,
    fixed_advantage: float | None = None,
) -> OPSAOutput:
    """Compute baseline OPSA advantages for one DP-local packed response batch.

    All valid response tokens are ranked together by current actor log-probability.
    The lowest ``floor(token_fraction * N)`` are selected (at least one when
    ``N > 0``). Unselected tokens receive zero advantage and a zero OPSA mask.
    """
    if not 0 < token_fraction <= 1:
        raise ValueError(f"OPSA token_fraction must be in (0, 1], got {token_fraction}.")
    if mode not in {"entropy", "fixed"}:
        raise ValueError(f"Baseline OPSA mode must be 'entropy' or 'fixed', got {mode!r}.")
    if mode == "entropy":
        if entropies is None:
            raise ValueError("OPSA entropy mode requires actor entropies.")
        if not advantage_min < advantage_max < 0:
            raise ValueError(
                "OPSA entropy advantages must satisfy advantage_min < advantage_max < 0, "
                f"got {advantage_min} and {advantage_max}."
            )
    elif fixed_advantage is None or fixed_advantage == 0:
        raise ValueError("OPSA fixed mode requires a non-zero fixed_advantage.")

    _validate_inputs(log_probs, loss_masks, entropies)

    if not log_probs:
        empty = torch.tensor(0.0)
        return OPSAOutput(
            advantages=[],
            loss_masks=[],
            metrics={
                "opsa/selected_fraction": empty,
                "opsa/selected_tokens": empty,
                "opsa/valid_tokens": empty,
                "opsa/advantage_mean": empty,
            },
        )

    device = log_probs[0].device
    flat_log_probs = torch.cat([x.detach().to(device=device, dtype=torch.float32) for x in log_probs])
    flat_valid_mask = torch.cat([x.to(device=device).bool() for x in loss_masks])
    flat_advantages = torch.zeros_like(flat_log_probs, dtype=torch.float32)
    flat_opsa_mask = torch.zeros_like(flat_log_probs, dtype=torch.float32)

    valid_indices = torch.nonzero(flat_valid_mask, as_tuple=False).flatten()
    valid_count = valid_indices.numel()
    if valid_count > 0:
        selected_count = min(max(1, math.floor(token_fraction * valid_count)), valid_count)
        order = torch.argsort(flat_log_probs[valid_indices], stable=True)
        selected_indices = valid_indices[order[:selected_count]]
        flat_opsa_mask[selected_indices] = 1.0

        if mode == "fixed":
            flat_advantages[selected_indices] = fixed_advantage
        else:
            flat_entropies = torch.cat(
                [x.detach().to(device=device, dtype=torch.float32) for x in entropies]
            )
            selected_entropies = flat_entropies[selected_indices]
            entropy_range = selected_entropies.max() - selected_entropies.min()
            if entropy_range <= 1e-12:
                entropy_rank = torch.ones_like(selected_entropies)
            else:
                entropy_rank = (selected_entropies - selected_entropies.min()) / entropy_range
            flat_advantages[selected_indices] = (
                advantage_max + (advantage_min - advantage_max) * entropy_rank
            )

    split_sizes = [x.numel() for x in log_probs]
    advantages = [
        x.reshape_as(log_prob).to(dtype=log_prob.dtype)
        for x, log_prob in zip(flat_advantages.split(split_sizes), log_probs, strict=True)
    ]
    opsa_masks = [
        x.reshape_as(loss_mask)
        for x, loss_mask in zip(flat_opsa_mask.split(split_sizes), loss_masks, strict=True)
    ]

    selected_tokens = flat_opsa_mask.sum()
    valid_tokens = flat_valid_mask.to(dtype=torch.float32).sum()
    selected_advantages = flat_advantages[flat_opsa_mask.bool()]
    advantage_mean = (
        selected_advantages.mean()
        if selected_advantages.numel() > 0
        else torch.zeros((), dtype=torch.float32, device=device)
    )
    metrics = {
        "opsa/selected_fraction": selected_tokens / torch.clamp_min(valid_tokens, 1),
        "opsa/selected_tokens": selected_tokens,
        "opsa/valid_tokens": valid_tokens,
        "opsa/advantage_mean": advantage_mean,
    }
    return OPSAOutput(advantages=advantages, loss_masks=opsa_masks, metrics=metrics)
