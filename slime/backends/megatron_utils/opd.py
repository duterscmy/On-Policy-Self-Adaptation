"""Token selection, objectives, and diagnostics for on-policy distillation."""

import math
from dataclasses import dataclass

import torch


_LOSS_TYPES = {"vanilla", "geometry_corrected", "bernoulli", "weighted"}
_TOKEN_FILTERS = {"all", "high_conf", "bottom_percent"}


def uses_direct_opd_objective(args) -> bool:
    """Whether OPD must be evaluated separately from the legacy advantage path."""
    return args.use_opd and (args.opd_loss_type != "vanilla" or args.opd_token_filter != "all")


def validate_opd_args(args) -> None:
    """Validate the parameterized OPD objective without changing legacy defaults."""
    if args.opd_loss_type not in _LOSS_TYPES:
        raise ValueError(f"Unsupported --opd-loss-type: {args.opd_loss_type!r}.")
    if args.opd_token_filter not in _TOKEN_FILTERS:
        raise ValueError(f"Unsupported --opd-token-filter: {args.opd_token_filter!r}.")
    if not 0 < args.opd_high_conf_threshold < 1:
        raise ValueError("--opd-high-conf-threshold must be in (0, 1).")
    if not 0 < args.opd_bottom_fraction <= 1:
        raise ValueError("--opd-bottom-fraction must be in (0, 1].")
    if not 0 <= args.opd_geometry_alpha <= 1:
        raise ValueError("--opd-geometry-alpha must be in [0, 1].")

    if not args.use_opd:
        if args.opd_loss_type != "vanilla" or args.opd_token_filter != "all":
            raise ValueError("Non-default OPD objectives and filters require --use-opd.")
        return

    if args.opd_kl_coef < 0:
        raise ValueError("--opd-kl-coef must be non-negative.")

    if uses_direct_opd_objective(args):
        if args.normalize_advantages:
            raise ValueError("Direct OPD objectives are incompatible with --normalize-advantages.")
        incompatible = {
            "--use-rollout-logprobs": args.use_rollout_logprobs,
            "--use-opsm": args.use_opsm,
            "--use-tis": args.use_tis,
            "--get-mismatch-metrics": args.get_mismatch_metrics,
            "--use-score-centering": getattr(args, "use_score_centering", False),
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                "Direct OPD objectives currently require an on-policy, unmodified policy loss; "
                f"disable {', '.join(enabled)}."
            )
        if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
            raise ValueError("Direct OPD objectives do not support a custom policy-loss reducer.")


@dataclass(frozen=True)
class OPDSelection:
    """Full-response selection masks and batch-level diagnostics."""

    loss_masks: list[torch.Tensor]
    metrics: dict[str, torch.Tensor]


def _validate_inputs(
    student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> None:
    if not (len(student_log_probs) == len(teacher_log_probs) == len(loss_masks)):
        raise ValueError("OPD requires one student tensor, teacher tensor, and loss mask per sample.")
    for index, (student, teacher, mask) in enumerate(
        zip(student_log_probs, teacher_log_probs, loss_masks, strict=True)
    ):
        if student.shape != teacher.shape or student.shape != mask.shape:
            raise ValueError(
                f"OPD tensor shape mismatch for sample {index}: student={tuple(student.shape)}, "
                f"teacher={tuple(teacher.shape)}, mask={tuple(mask.shape)}."
            )


def compute_opd_selection(
    student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    *,
    token_filter: str,
    high_conf_threshold: float,
    bottom_fraction: float,
    geometry_alpha: float,
) -> OPDSelection:
    """Build a float selection mask and token-level diagnostics for one DP-local batch."""
    if token_filter not in _TOKEN_FILTERS:
        raise ValueError(f"Unsupported OPD token filter: {token_filter!r}.")
    _validate_inputs(student_log_probs, teacher_log_probs, loss_masks)

    if not student_log_probs:
        zero = torch.tensor(0.0)
        return OPDSelection([], {"opd/selected_fraction": zero, "opd/valid_tokens": zero})

    device = student_log_probs[0].device
    student_lp = torch.cat([x.detach().to(device=device, dtype=torch.float32) for x in student_log_probs])
    teacher_lp = torch.cat([x.detach().to(device=device, dtype=torch.float32) for x in teacher_log_probs])
    valid = torch.cat([x.to(device=device).bool() for x in loss_masks])
    selected = torch.zeros_like(student_lp, dtype=torch.float32)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()

    if token_filter == "all":
        selected[valid_indices] = 1.0
    elif token_filter == "high_conf":
        keep = student_lp[valid_indices].exp() > high_conf_threshold
        selected[valid_indices[keep]] = 1.0
    elif valid_indices.numel() > 0:
        selected_count = min(max(1, math.floor(bottom_fraction * valid_indices.numel())), valid_indices.numel())
        order = torch.argsort(student_lp[valid_indices], stable=True)
        selected[valid_indices[order[:selected_count]]] = 1.0

    split_sizes = [x.numel() for x in student_log_probs]
    masks = [
        chunk.reshape_as(mask)
        for chunk, mask in zip(selected.split(split_sizes), loss_masks, strict=True)
    ]

    valid_float = valid.to(torch.float32)
    selected_bool = selected.bool()
    valid_count = valid_float.sum()
    selected_count = selected.sum()
    student_prob = student_lp.exp().clamp(0.0, 1.0)
    teacher_prob = teacher_lp.exp().clamp(0.0, 1.0)
    advantage = teacher_lp - student_lp
    prob_gap = teacher_prob - student_prob

    def valid_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * valid_float).sum() / torch.clamp_min(valid_count, 1)

    def selected_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * selected).sum() / torch.clamp_min(selected_count, 1)

    one_minus_p = (1.0 - student_prob).clamp_min(1e-6)
    metrics = {
        "opd/selected_fraction": selected_count / torch.clamp_min(valid_count, 1),
        "opd/selected_tokens": selected_count,
        "opd/valid_tokens": valid_count,
        "opd/student_prob_mean": valid_mean(student_prob),
        "opd/teacher_prob_mean": valid_mean(teacher_prob),
        "opd/advantage_mean": valid_mean(advantage),
        "opd/prob_gap_mean": valid_mean(prob_gap),
        "opd/selected_student_prob_mean": selected_mean(student_prob),
        "opd/selected_teacher_prob_mean": selected_mean(teacher_prob),
        "opd/selected_advantage_mean": selected_mean(advantage),
        "opd/proxy_vanilla_abs_mean": valid_mean(advantage.abs() * one_minus_p),
        "opd/proxy_corrected_abs_mean": valid_mean(advantage.abs()),
        "opd/proxy_bernoulli_abs_mean": valid_mean(prob_gap.abs()),
        "opd/proxy_weighted_abs_mean": valid_mean(advantage.abs() * one_minus_p.pow(1.0 - geometry_alpha)),
    }
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        in_bin = valid & (student_prob >= lower)
        in_bin &= student_prob <= upper if index == 9 else student_prob < upper
        bin_count = in_bin.to(torch.float32).sum()
        selected_in_bin = (in_bin & selected_bool).to(torch.float32).sum()
        metrics[f"opd/pS_bin_{index}_fraction"] = bin_count / torch.clamp_min(valid_count, 1)
        metrics[f"opd/pS_bin_{index}_selected_fraction"] = selected_in_bin / torch.clamp_min(bin_count, 1)

    return OPDSelection(loss_masks=masks, metrics={key: value.detach() for key, value in metrics.items()})


def compute_opd_token_loss(
    current_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    loss_type: str,
    coef: float,
    geometry_alpha: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return an unreduced direct OPD loss over sampled response tokens."""
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"Unsupported OPD loss type: {loss_type!r}.")

    current_lp = current_log_probs.to(torch.float32)
    student_lp = student_log_probs.detach().to(device=current_lp.device, dtype=torch.float32)
    teacher_lp = teacher_log_probs.detach().to(device=current_lp.device, dtype=torch.float32)
    advantage = teacher_lp - student_lp

    if loss_type == "vanilla":
        token_loss = -advantage * current_lp
    elif loss_type == "geometry_corrected":
        current_prob = current_lp.exp().clamp(min=eps, max=1.0 - eps)
        current_log_odds = current_lp - torch.log1p(-current_prob)
        token_loss = -advantage * current_log_odds
    elif loss_type == "weighted":
        student_prob = student_lp.exp().clamp(min=0.0, max=1.0 - eps)
        weight = (1.0 - student_prob).clamp_min(eps).pow(-geometry_alpha)
        token_loss = -weight * advantage * current_lp
    else:
        current_prob = current_lp.exp().clamp(min=eps, max=1.0 - eps)
        teacher_prob = teacher_lp.exp().clamp(min=eps, max=1.0 - eps)
        token_loss = -teacher_prob * current_lp - (1.0 - teacher_prob) * torch.log1p(-current_prob)

    return coef * token_loss
