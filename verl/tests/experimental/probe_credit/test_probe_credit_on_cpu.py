import numpy as np
import pytest
import torch

from verl.experimental.probe_credit.probe_credit import (
    apply_probe_credit_redistribution,
    build_probe_token_correction,
    compute_group_relative_probe_advantages,
    compute_probe_pseudo_rewards,
    compute_probe_temporal_returns,
)


def test_exact_probe_progress_and_discounted_returns_example():
    values = torch.tensor([[0.0, 0.0, 0.0, 0.75, 0.75]])
    rewards = compute_probe_pseudo_rewards(values)
    returns = compute_probe_temporal_returns(rewards, rho=0.5)

    torch.testing.assert_close(rewards, torch.tensor([[0.0, 0.0, 0.75, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[0.1875, 0.375, 0.75, 0.0]]))


def test_temporal_returns_rho_endpoints_and_negative_progress():
    rewards = torch.tensor([[0.5, -0.25, 0.75, -1.0]])
    torch.testing.assert_close(compute_probe_temporal_returns(rewards, rho=0.0), rewards)
    torch.testing.assert_close(
        compute_probe_temporal_returns(rewards, rho=1.0),
        torch.tensor([[0.0, -0.5, -0.25, -1.0]]),
    )


@pytest.mark.parametrize("group_size", [2, 8])
def test_group_relative_advantages_match_torch_sample_std(group_size):
    base = torch.arange(group_size, dtype=torch.float32)
    returns = torch.stack([base, base * 2, -base, base + 3], dim=-1)
    actual, metrics = compute_group_relative_probe_advantages(
        returns, np.array(["prompt"] * group_size, dtype=object), norm_by_std=True, epsilon=1e-6
    )
    expected = (returns - returns.mean(dim=0)) / (returns.std(dim=0, correction=1) + 1e-6)

    torch.testing.assert_close(actual, expected)
    assert metrics["probe_credit/degenerate_rate"] == 0.0


def test_group_relative_advantages_zero_degenerate_segments():
    returns = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 3.0, 8.0], [9.0, 9.0, 9.0, 9.0]])
    groups = np.array(["pair", "pair", "singleton"], dtype=object)

    actual, metrics = compute_group_relative_probe_advantages(returns, groups, epsilon=1e-6)

    assert torch.equal(actual[:, 0], torch.zeros(3))
    assert torch.equal(actual[:, 2], torch.zeros(3))
    assert torch.equal(actual[2], torch.zeros(4))
    assert metrics["probe_credit/degenerate_rate"] == pytest.approx(8 / 12)


def test_token_correction_has_zero_masked_mass_and_exact_tail_padding_zeros():
    segment_advantages = torch.tensor([[1.0, 2.0, -1.0, 4.0], [-2.0, 3.0, 5.0, -1.0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    correction, metrics = build_probe_token_correction(segment_advantages, response_mask)

    torch.testing.assert_close((correction * response_mask).sum(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.equal(correction[0, 9:], torch.zeros(3))  # floor(.9 * 10) through padding
    assert torch.equal(correction[1, 6:], torch.zeros(6))  # floor(.9 * 7) through padding
    assert metrics["probe_credit/zero_mass_residual_max"] <= 1e-6


def test_apply_preserves_terminal_anchor_rewards_and_grpo_returns_contract():
    terminal = torch.tensor([[2.0, 2.0, 0.0], [-1.0, -1.0, 0.0]])
    correction = torch.tensor([[1.0, -1.0, 0.0], [0.5, -0.5, 0.0]])
    scores = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    rewards = scores.clone()
    batch = {
        "advantages": terminal.clone(),
        "returns": terminal.clone(),
        "token_level_scores": scores,
        "token_level_rewards": rewards,
    }

    metrics = apply_probe_credit_redistribution(batch, correction, coef=0.25, enable=True)

    assert torch.equal(batch["terminal_advantages"], terminal)
    torch.testing.assert_close(batch["advantages"], terminal + 0.25 * correction)
    assert torch.equal(batch["returns"], batch["advantages"])
    assert torch.equal(batch["token_level_scores"], scores)
    assert torch.equal(batch["token_level_rewards"], rewards)
    assert "probe_credit/final_advantage_mean" in metrics


def test_disabled_and_zero_coefficient_leave_advantages_equal_to_baseline():
    baseline = torch.tensor([[1.0, -1.0]])
    correction = torch.tensor([[3.0, -3.0]])
    disabled = {"advantages": baseline.clone(), "returns": baseline.clone()}
    zero_coef = {"advantages": baseline.clone(), "returns": baseline.clone()}

    apply_probe_credit_redistribution(disabled, correction, coef=1.0, enable=False)
    apply_probe_credit_redistribution(zero_coef, correction, coef=0.0, enable=True)

    assert set(disabled) == {"advantages", "returns"}
    assert torch.equal(disabled["advantages"], baseline)
    assert torch.equal(zero_coef["advantages"], baseline)
    assert torch.equal(zero_coef["returns"], baseline)
