import math

import pytest
import torch

from slime.backends.megatron_utils.opd import compute_opd_selection, compute_opd_token_loss


def _sampled_logit_gradient(loss_type: str, *, alpha: float = 0.5):
    logits = torch.tensor([0.2, -0.4, 1.1], dtype=torch.float64, requires_grad=True)
    current_log_prob = torch.log_softmax(logits, dim=0)[1:2]
    student_log_prob = current_log_prob.detach().clone()
    teacher_log_prob = torch.tensor([math.log(0.35)], dtype=torch.float64)
    loss = compute_opd_token_loss(
        current_log_prob,
        student_log_prob,
        teacher_log_prob,
        loss_type=loss_type,
        coef=0.7,
        geometry_alpha=alpha,
    ).sum()
    loss.backward()
    return logits.grad[1], student_log_prob.exp().item(), teacher_log_prob.exp().item(), (
        teacher_log_prob - student_log_prob
    ).item()


@pytest.mark.parametrize(
    ("loss_type", "alpha"),
    [("vanilla", 0.5), ("geometry_corrected", 0.5), ("bernoulli", 0.5), ("weighted", 0.4)],
)
def test_opd_sampled_logit_gradients_match_analytic_forms(loss_type, alpha):
    gradient, student_prob, teacher_prob, advantage = _sampled_logit_gradient(loss_type, alpha=alpha)
    if loss_type == "vanilla":
        expected = -0.7 * advantage * (1.0 - student_prob)
    elif loss_type == "geometry_corrected":
        expected = -0.7 * advantage
    elif loss_type == "bernoulli":
        expected = 0.7 * (student_prob - teacher_prob)
    else:
        expected = -0.7 * advantage * (1.0 - student_prob) ** (1.0 - alpha)
    assert gradient.item() == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_high_conf_filter_is_strict_and_mask_stays_float():
    student = [torch.log(torch.tensor([0.2, 0.5, 0.5001, 0.9]))]
    teacher = [torch.log(torch.tensor([0.3, 0.4, 0.6, 0.8]))]
    masks = [torch.ones(4, dtype=torch.int64)]
    out = compute_opd_selection(
        student,
        teacher,
        masks,
        token_filter="high_conf",
        high_conf_threshold=0.5,
        bottom_fraction=0.2,
        geometry_alpha=0.5,
    )
    assert out.loss_masks[0].is_floating_point()
    assert out.loss_masks[0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert out.metrics["opd/selected_fraction"].item() == pytest.approx(0.5)


def test_bottom_percent_ranks_valid_tokens_across_samples():
    student = [torch.tensor([-0.1, -4.0, -0.2]), torch.tensor([-3.0, -2.0])]
    teacher = [torch.zeros(3), torch.zeros(2)]
    masks = [torch.tensor([1, 0, 1]), torch.ones(2, dtype=torch.int64)]
    out = compute_opd_selection(
        student,
        teacher,
        masks,
        token_filter="bottom_percent",
        high_conf_threshold=0.5,
        bottom_fraction=0.5,
        geometry_alpha=0.5,
    )
    # Four valid tokens -> floor(0.5 * 4) = 2; the masked -4 token is ignored.
    assert out.loss_masks[0].tolist() == [0.0, 0.0, 0.0]
    assert out.loss_masks[1].tolist() == [1.0, 1.0]


def test_objectives_remain_finite_near_probability_one():
    current = torch.tensor([-1e-8], requires_grad=True)
    student = current.detach().clone()
    teacher = torch.tensor([math.log(0.8)])
    losses = [
        compute_opd_token_loss(
            current,
            student,
            teacher,
            loss_type=loss_type,
            coef=1.0,
            geometry_alpha=0.5,
        )
        for loss_type in ("geometry_corrected", "bernoulli", "weighted")
    ]
    assert all(torch.isfinite(loss).all() for loss in losses)
