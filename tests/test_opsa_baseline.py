import torch

from slime.backends.megatron_utils.opsa import compute_opsa


def test_fixed_opsa_selects_lowest_logp_tokens_across_batch():
    log_probs = [torch.tensor([-0.1, -3.0, -0.2]), torch.tensor([-2.0, -0.3])]
    masks = [torch.ones(3), torch.ones(2)]
    out = compute_opsa(
        log_probs,
        masks,
        token_fraction=0.4,
        mode="fixed",
        fixed_advantage=-0.5,
    )
    assert sum(int(mask.sum().item()) for mask in out.loss_masks) == 2
    assert out.loss_masks[0].tolist() == [0.0, 1.0, 0.0]
    assert out.loss_masks[1].tolist() == [1.0, 0.0]
    assert out.advantages[0].tolist() == [0.0, -0.5, 0.0]
    assert out.advantages[1].tolist() == [-0.5, 0.0]


def test_opsa_loss_mask_stays_float_with_integer_response_mask():
    log_probs = [torch.tensor([-0.1, -3.0, -0.2, -2.0])]
    masks = [torch.ones(4, dtype=torch.int64)]
    out = compute_opsa(
        log_probs,
        masks,
        token_fraction=0.5,
        mode="fixed",
        fixed_advantage=-0.5,
    )
    assert out.loss_masks[0].is_floating_point()
    assert out.loss_masks[0].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_entropy_opsa_maps_higher_entropy_to_more_negative_advantage():
    log_probs = [torch.tensor([-3.0, -2.0, -0.1, -0.2])]
    masks = [torch.ones(4)]
    entropies = [torch.tensor([2.0, 1.0, 0.1, 0.1])]
    out = compute_opsa(
        log_probs,
        masks,
        token_fraction=0.5,
        mode="entropy",
        entropies=entropies,
        advantage_min=-1.0,
        advantage_max=-0.5,
    )
    assert out.loss_masks[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert torch.allclose(out.advantages[0][:2], torch.tensor([-1.0, -0.5]))
