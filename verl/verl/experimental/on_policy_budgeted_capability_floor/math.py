"""Pure OBCF floor and on-policy capability-advantage mathematics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CapabilityAdvantageResult:
    q_current: torch.Tensor
    deficit: torch.Tensor
    active_group: torch.Tensor
    centered_prefix_reward: torch.Tensor
    token_advantage: torch.Tensor
    observed_constraint: torch.Tensor
    mixed_group_fraction: torch.Tensor
    all_zero_group_fraction: torch.Tensor
    all_one_group_fraction: torch.Tensor
    nonzero_gradient_group_fraction: torch.Tensor


@dataclass(frozen=True)
class FloorActionabilityReport:
    current_rollouts_per_prompt: int
    protected_prompt_count: int
    actionable_prompt_count: int
    inert_prompt_count: int
    inert_prompt_fraction: float
    minimum_positive_empirical_rate: float
    by_base_success_count: dict[int, dict[str, float | int]]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def compute_capability_floor(
    *,
    base_success_count: int,
    base_rollout_count: int,
    tolerance_count: int,
) -> float:
    """Return the conservative Base capability floor in exact count space."""
    if not _is_int(base_rollout_count) or base_rollout_count <= 0:
        raise ValueError("base_rollout_count must be a positive integer")
    if (
        not _is_int(base_success_count)
        or base_success_count < 0
        or base_success_count > base_rollout_count
    ):
        raise ValueError("base_success_count must be in [0, base_rollout_count]")
    if (
        not _is_int(tolerance_count)
        or tolerance_count < 0
        or tolerance_count > base_rollout_count
    ):
        raise ValueError("tolerance_count must be in [0, base_rollout_count]")
    return max(base_success_count - tolerance_count, 0) / base_rollout_count


def _validate_current_rollout_count(current_rollouts_per_prompt: int) -> None:
    if (
        not _is_int(current_rollouts_per_prompt)
        or current_rollouts_per_prompt <= 0
    ):
        raise ValueError("current_rollouts_per_prompt must be a positive integer")


def floor_is_actionable(
    *,
    capability_floor: float,
    current_rollouts_per_prompt: int,
) -> bool:
    """Return whether a mixed empirical group can strictly violate the floor."""
    _validate_current_rollout_count(current_rollouts_per_prompt)
    if (
        isinstance(capability_floor, bool)
        or not isinstance(capability_floor, (int, float))
        or not math.isfinite(float(capability_floor))
        or not 0.0 <= float(capability_floor) <= 1.0
    ):
        raise ValueError("capability_floor must be finite and within [0, 1]")
    return float(capability_floor) > (1.0 / current_rollouts_per_prompt)


def summarize_floor_actionability(
    *,
    cache_rows: Iterable[Mapping[str, Any]],
    current_rollouts_per_prompt: int,
) -> FloorActionabilityReport:
    """Summarize protected floors that can produce a nonzero mixed-group gradient."""
    _validate_current_rollout_count(current_rollouts_per_prompt)
    rows = list(cache_rows)
    strata: dict[int, dict[str, int | float]] = defaultdict(
        lambda: {
            "protected_prompt_count": 0,
            "actionable_prompt_count": 0,
            "inert_prompt_count": 0,
        }
    )
    actionable_count = 0
    for row in rows:
        try:
            success_count = int(row["base_prefix_success_count"])
            capability_floor = float(row["capability_floor"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cache row is missing valid floor actionability fields") from error
        if isinstance(row.get("base_prefix_success_count"), bool) or success_count < 0:
            raise ValueError("base_prefix_success_count must be a nonnegative integer")
        float_actionable = floor_is_actionable(
            capability_floor=capability_floor,
            current_rollouts_per_prompt=current_rollouts_per_prompt,
        )
        if "floor_count" in row or "base_rollout_count" in row:
            try:
                floor_count = int(row["floor_count"])
                base_rollout_count = int(row["base_rollout_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("cache row count-space floor fields must be valid integers") from error
            if (
                isinstance(row["floor_count"], bool)
                or isinstance(row["base_rollout_count"], bool)
                or floor_count < 0
                or base_rollout_count <= 0
                or floor_count > base_rollout_count
            ):
                raise ValueError("cache row count-space floor fields are invalid")
            count_actionable = (
                floor_count * current_rollouts_per_prompt > base_rollout_count
            )
            if count_actionable != float_actionable or not math.isclose(
                capability_floor,
                floor_count / base_rollout_count,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("count-space and float-space capability floors disagree")
            actionable = count_actionable
        else:
            actionable = float_actionable
        stratum = strata[success_count]
        stratum["protected_prompt_count"] = int(stratum["protected_prompt_count"]) + 1
        bucket = "actionable_prompt_count" if actionable else "inert_prompt_count"
        stratum[bucket] = int(stratum[bucket]) + 1
        actionable_count += int(actionable)

    by_success: dict[int, dict[str, float | int]] = {}
    for success_count, values in sorted(strata.items()):
        protected = int(values["protected_prompt_count"])
        inert = int(values["inert_prompt_count"])
        by_success[success_count] = {
            "protected_prompt_count": protected,
            "actionable_prompt_count": int(values["actionable_prompt_count"]),
            "inert_prompt_count": inert,
            "inert_prompt_fraction": inert / protected,
        }
    protected_count = len(rows)
    inert_count = protected_count - actionable_count
    return FloorActionabilityReport(
        current_rollouts_per_prompt=current_rollouts_per_prompt,
        protected_prompt_count=protected_count,
        actionable_prompt_count=actionable_count,
        inert_prompt_count=inert_count,
        inert_prompt_fraction=(inert_count / protected_count if protected_count else 0.0),
        minimum_positive_empirical_rate=1.0 / current_rollouts_per_prompt,
        by_base_success_count=by_success,
    )


def _zero_scalar(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def compute_capability_advantage(
    *,
    prefix_rewards: torch.Tensor,
    group_ids: torch.Tensor,
    capability_floors: torch.Tensor,
    response_mask: torch.Tensor,
    reference_budget: int,
) -> CapabilityAdvantageResult:
    """Compute gated, group-centered, prefix-only OBCF token advantages."""
    if prefix_rewards.ndim != 1:
        raise ValueError("prefix_rewards must be one-dimensional")
    if group_ids.ndim != 1 or group_ids.shape != prefix_rewards.shape:
        raise ValueError("group_ids must match prefix_rewards")
    if response_mask.ndim != 2 or response_mask.shape[0] != prefix_rewards.shape[0]:
        raise ValueError("response_mask must have one row per prefix reward")
    if capability_floors.ndim != 1:
        raise ValueError("capability_floors must be one-dimensional")
    if not _is_int(reference_budget) or not 0 < reference_budget <= response_mask.shape[1]:
        raise ValueError("reference_budget must be within the response-mask width")
    if prefix_rewards.dtype.is_complex:
        raise ValueError("prefix_rewards must use a real dtype")
    if capability_floors.dtype.is_complex:
        raise ValueError("capability_floors must use a real dtype")
    if response_mask.dtype.is_complex:
        raise ValueError("response_mask must use a real dtype")
    if group_ids.dtype == torch.bool or group_ids.dtype.is_floating_point or group_ids.dtype.is_complex:
        raise ValueError("group_ids must use an integer dtype")
    if not bool(torch.isfinite(prefix_rewards).all().item()):
        raise ValueError("prefix_rewards must be finite")
    if not bool(((prefix_rewards == 0) | (prefix_rewards == 1)).all().item()):
        raise ValueError("prefix_rewards must be binary")
    if not bool(torch.isfinite(capability_floors).all().item()):
        raise ValueError("capability_floors must be finite")
    if not bool(((capability_floors >= 0) & (capability_floors <= 1)).all().item()):
        raise ValueError("capability_floors must be within [0, 1]")
    if not bool(((response_mask == 0) | (response_mask == 1)).all().item()):
        raise ValueError("response_mask must be binary")
    if not (
        prefix_rewards.device == group_ids.device
        == capability_floors.device
        == response_mask.device
    ):
        raise ValueError("all capability tensors must be on the same device")

    rollout_count = prefix_rewards.shape[0]
    group_count = capability_floors.numel()
    device = prefix_rewards.device
    rewards = prefix_rewards.to(dtype=torch.float32)
    floors = capability_floors.to(dtype=torch.float32)

    if rollout_count == 0:
        if group_count != 0 or group_ids.numel() != 0:
            raise ValueError("empty rewards require empty group IDs and capability floors")
        zero = _zero_scalar(device)
        return CapabilityAdvantageResult(
            q_current=torch.empty(0, dtype=torch.float32, device=device),
            deficit=torch.empty(0, dtype=torch.float32, device=device),
            active_group=torch.empty(0, dtype=torch.bool, device=device),
            centered_prefix_reward=torch.empty(0, dtype=torch.float32, device=device),
            token_advantage=torch.zeros_like(response_mask, dtype=torch.float32),
            observed_constraint=zero,
            mixed_group_fraction=zero,
            all_zero_group_fraction=zero,
            all_one_group_fraction=zero,
            nonzero_gradient_group_fraction=zero,
        )
    if group_count == 0:
        raise ValueError("capability_floors must be nonempty for protected rollouts")
    ids = group_ids.to(dtype=torch.long)
    if bool((ids < 0).any().item()):
        raise ValueError("group_ids must be nonnegative and contiguous")
    expected_ids = torch.arange(group_count, device=device)
    present_ids = torch.unique(ids, sorted=True)
    if not torch.equal(present_ids, expected_ids):
        raise ValueError("group_ids must be contiguous from zero and match capability floors")
    counts = torch.bincount(ids, minlength=group_count)
    if not bool((counts == counts[0]).all().item()):
        raise ValueError("protected groups must have equal rollout counts")

    reward_sums = torch.zeros(group_count, dtype=torch.float32, device=device)
    reward_sums.scatter_add_(0, ids, rewards)
    q_current = reward_sums / counts.to(dtype=torch.float32)
    deficit = torch.relu(floors - q_current)
    active_group = deficit > 0
    centered = rewards - q_current[ids]
    gated_centered = centered * active_group[ids].to(dtype=torch.float32)

    prefix_positions = torch.arange(response_mask.shape[1], device=device) < reference_budget
    prefix_mask = response_mask.to(dtype=torch.bool) & prefix_positions.unsqueeze(0)
    token_advantage = gated_centered.unsqueeze(1) * prefix_mask.to(dtype=torch.float32)

    mixed = (q_current > 0) & (q_current < 1)
    all_zero = q_current == 0
    all_one = q_current == 1
    rollout_has_gradient = torch.any(token_advantage != 0, dim=1)
    gradient_counts = torch.zeros(group_count, dtype=torch.long, device=device)
    gradient_counts.scatter_add_(0, ids, rollout_has_gradient.to(dtype=torch.long))
    nonzero_gradient = gradient_counts > 0

    return CapabilityAdvantageResult(
        q_current=q_current,
        deficit=deficit,
        active_group=active_group,
        centered_prefix_reward=centered,
        token_advantage=token_advantage,
        observed_constraint=deficit.mean(),
        mixed_group_fraction=mixed.float().mean(),
        all_zero_group_fraction=all_zero.float().mean(),
        all_one_group_fraction=all_one.float().mean(),
        nonzero_gradient_group_fraction=nonzero_gradient.float().mean(),
    )
