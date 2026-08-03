import math

import pytest
import torch

from verl.experimental.on_policy_budgeted_capability_floor.math import (
    compute_capability_advantage,
    compute_capability_floor,
)
from verl.trainer.config import OnPolicyBudgetedCapabilityFloorConfig


def test_config_scientific_defaults_are_exact_and_inert():
    config = OnPolicyBudgetedCapabilityFloorConfig()
    config.validate()

    assert vars(config) == {
        "_target_": "",
        "mode": "off",
        "cache_path": None,
        "reference_budget": 2048,
        "base_rollouts_per_prompt": 8,
        "support_threshold": 2,
        "reference_tolerance_count": 1,
        "delta": 0.05,
        "update_interval": 1,
        "lambda_init": 0.0,
        "lambda_max": 10.0,
        "dual_lr": 0.01,
        "dual_ema_beta": 0.9,
        "seed": 20260803,
        "strict": True,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "active"),
        ("reference_budget", 0),
        ("reference_budget", True),
        ("base_rollouts_per_prompt", 0),
        ("support_threshold", 0),
        ("support_threshold", 9),
        ("reference_tolerance_count", -1),
        ("reference_tolerance_count", 9),
        ("update_interval", 0),
        ("delta", -0.1),
        ("delta", float("nan")),
        ("delta", 1.1),
        ("delta", True),
        ("delta", "bad"),
        ("lambda_init", -0.1),
        ("lambda_init", True),
        ("lambda_max", -0.1),
        ("lambda_max", float("inf")),
        ("lambda_max", None),
        ("dual_lr", -0.1),
        ("dual_lr", True),
        ("dual_ema_beta", -0.1),
        ("dual_ema_beta", 1.0),
        ("dual_ema_beta", float("nan")),
        ("seed", True),
        ("cache_path", ""),
        ("strict", False),
    ],
)
def test_config_invalid_values_fail_closed(field, value):
    config = OnPolicyBudgetedCapabilityFloorConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        config.validate()


def test_config_lambda_max_covers_initial_value():
    with pytest.raises(ValueError, match="lambda_max"):
        OnPolicyBudgetedCapabilityFloorConfig(lambda_init=2.0, lambda_max=1.0).validate()


def test_capability_floor_uses_count_tolerance():
    assert compute_capability_floor(
        base_success_count=2,
        base_rollout_count=8,
        tolerance_count=1,
    ) == pytest.approx(1 / 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_success_count": -1, "base_rollout_count": 8, "tolerance_count": 1},
        {"base_success_count": 9, "base_rollout_count": 8, "tolerance_count": 1},
        {"base_success_count": 1, "base_rollout_count": 0, "tolerance_count": 1},
        {"base_success_count": 1, "base_rollout_count": 8, "tolerance_count": -1},
        {"base_success_count": 1, "base_rollout_count": 8, "tolerance_count": 9},
        {"base_success_count": True, "base_rollout_count": 8, "tolerance_count": 1},
    ],
)
def test_capability_floor_rejects_invalid_counts(kwargs):
    with pytest.raises(ValueError):
        compute_capability_floor(**kwargs)


def _compute(rewards, floors, *, width=4, budget=2, mask=None):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    group_size = rewards.numel() // len(floors) if floors else 0
    group_ids = torch.arange(len(floors), dtype=torch.long).repeat_interleave(group_size)
    if mask is None:
        mask = torch.ones((rewards.numel(), width), dtype=torch.bool)
    return compute_capability_advantage(
        prefix_rewards=rewards,
        group_ids=group_ids,
        capability_floors=torch.tensor(floors),
        response_mask=mask,
        reference_budget=budget,
    )


def test_mixed_group_is_mean_centered_prefix_only_and_not_standardized():
    result = _compute([0, 1], [0.75])

    assert torch.equal(result.q_current, torch.tensor([0.5]))
    assert torch.equal(result.deficit, torch.tensor([0.25]))
    assert torch.equal(result.centered_prefix_reward, torch.tensor([-0.5, 0.5]))
    assert torch.equal(
        result.token_advantage,
        torch.tensor([[-0.5, -0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]),
    )
    assert result.token_advantage[:, :2].sum(dim=0).tolist() == pytest.approx([0.0, 0.0])
    assert result.observed_constraint.item() == pytest.approx(0.25)
    assert result.active_group.item()
    assert result.mixed_group_fraction.item() == pytest.approx(1.0)
    assert result.nonzero_gradient_group_fraction.item() == pytest.approx(1.0)


def test_all_zero_active_group_reports_deficit_but_has_exactly_zero_gradient():
    result = _compute([0, 0], [0.5])

    assert result.deficit.item() == pytest.approx(0.5)
    assert result.active_group.item()
    assert torch.count_nonzero(result.token_advantage).item() == 0
    assert result.all_zero_group_fraction.item() == pytest.approx(1.0)
    assert result.nonzero_gradient_group_fraction.item() == pytest.approx(0.0)


def test_all_one_and_inactive_groups_have_zero_capability_advantage():
    result = _compute([1, 1, 0, 1], [1.0, 0.25])

    assert torch.equal(result.active_group, torch.tensor([False, False]))
    assert torch.count_nonzero(result.token_advantage).item() == 0
    assert result.all_one_group_fraction.item() == pytest.approx(0.5)
    assert result.mixed_group_fraction.item() == pytest.approx(0.5)


def test_padding_masks_capability_advantage():
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    result = _compute([0, 1], [1.0], width=3, budget=3, mask=mask)

    assert torch.equal(
        result.token_advantage,
        torch.tensor([[-0.5, 0.0, -0.5], [0.5, 0.5, 0.0]]),
    )


def test_empty_protected_set_returns_well_typed_zero_metrics():
    result = compute_capability_advantage(
        prefix_rewards=torch.empty(0),
        group_ids=torch.empty(0, dtype=torch.long),
        capability_floors=torch.empty(0),
        response_mask=torch.empty((0, 4), dtype=torch.bool),
        reference_budget=2,
    )

    assert result.token_advantage.shape == (0, 4)
    assert result.q_current.numel() == 0
    assert result.observed_constraint.item() == 0.0
    assert result.mixed_group_fraction.item() == 0.0


@pytest.mark.parametrize(
    ("rewards", "group_ids", "floors", "mask", "budget", "match"),
    [
        ([0.0, 0.5], [0, 0], [1.0], [[1], [1]], 1, "binary"),
        ([0.0, math.nan], [0, 0], [1.0], [[1], [1]], 1, "finite"),
        ([0.0, 1.0], [0, 2], [1.0, 1.0], [[1], [1]], 1, "contiguous"),
        ([0.0, 1.0, 0.0], [0, 0, 1], [1.0, 1.0], [[1], [1], [1]], 1, "equal"),
        ([0.0, 1.0], [0, 0], [], [[1], [1]], 1, "floors"),
        ([0.0, 1.0], [0, 0], [1.1], [[1], [1]], 1, "floors"),
        ([0.0, 1.0], [0, 0], [1.0], [[1], [2]], 1, "binary"),
        ([0.0, 1.0], [0, 0], [1.0], [[1], [1]], 2, "budget"),
    ],
)
def test_invalid_advantage_inputs_fail_closed(rewards, group_ids, floors, mask, budget, match):
    with pytest.raises(ValueError, match=match):
        compute_capability_advantage(
            prefix_rewards=torch.tensor(rewards),
            group_ids=torch.tensor(group_ids),
            capability_floors=torch.tensor(floors),
            response_mask=torch.tensor(mask),
            reference_budget=budget,
        )


@pytest.mark.parametrize("field", ["prefix_rewards", "group_ids", "capability_floors", "response_mask"])
def test_complex_advantage_inputs_fail_closed(field):
    kwargs = {
        "prefix_rewards": torch.tensor([0.0, 1.0]),
        "group_ids": torch.tensor([0, 0]),
        "capability_floors": torch.tensor([1.0]),
        "response_mask": torch.ones((2, 1)),
        "reference_budget": 1,
    }
    kwargs[field] = kwargs[field].to(dtype=torch.complex64)

    with pytest.raises(ValueError, match="dtype"):
        compute_capability_advantage(**kwargs)
