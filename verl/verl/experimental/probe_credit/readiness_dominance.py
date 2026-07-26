"""Pure PyTorch direct readiness dominance mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DominanceResult:
    """Direct pairwise edges and their per-trajectory set classification."""

    dominance_matrix: torch.Tensor
    eligible_mask: torch.Tensor
    frontier_mask: torch.Tensor
    dominated_mask: torch.Tensor
    group_has_dominance: torch.Tensor


def _require_bool_vector(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape ({batch_size},)")
    if tensor.dtype is not torch.bool:
        raise ValueError(f"{name} must have boolean dtype")


def _group_indices(group_ids: Sequence[object]) -> list[list[int]]:
    groups: list[tuple[object, list[int]]] = []
    for index, group_id in enumerate(group_ids):
        for existing_id, indices in groups:
            if group_id == existing_id:
                indices.append(index)
                break
        else:
            groups.append((group_id, [index]))
    return [indices for _, indices in groups]


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


@torch.no_grad()
def compute_horizon_readiness_metrics(
    probe_values: torch.Tensor,
    valid_mask: torch.Tensor,
    terminal_success: torch.Tensor,
    absolute_horizons: Sequence[int],
) -> dict[str, float]:
    """Summarize readiness on each horizon's active terminal-success subset."""
    if not isinstance(probe_values, torch.Tensor) or probe_values.ndim != 2:
        raise ValueError("probe_values must be a two-dimensional tensor")
    if not probe_values.is_floating_point():
        raise ValueError("probe_values must have floating dtype")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != probe_values.shape:
        raise ValueError("valid_mask and probe_values must have the same shape")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must have boolean dtype")
    batch_size, position_count = probe_values.shape
    if (
        not isinstance(terminal_success, torch.Tensor)
        or terminal_success.shape != (batch_size,)
    ):
        raise ValueError(f"terminal_success must have shape [{batch_size}]")
    if terminal_success.dtype is not torch.bool:
        raise ValueError("terminal_success must have boolean dtype")
    if (
        valid_mask.device != probe_values.device
        or terminal_success.device != probe_values.device
    ):
        raise ValueError("all readiness tensors must be on the same device")
    try:
        horizons = tuple(absolute_horizons)
    except TypeError as exc:
        raise ValueError("absolute_horizons must be a sequence") from exc
    if len(horizons) != position_count:
        raise ValueError("absolute_horizons length must match probe_values columns")
    if any(
        not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("absolute_horizons must contain positive integers")
    if any(left >= right for left, right in zip(horizons, horizons[1:], strict=False)):
        raise ValueError("absolute_horizons must be strictly increasing")
    if not bool(torch.isfinite(probe_values).all().item()):
        raise ValueError("probe_values must be finite")
    if not bool(((probe_values >= 0.0) & (probe_values <= 1.0)).all().item()):
        raise ValueError("probe_values must be in [0, 1]")
    if bool((valid_mask.any(dim=1) & ~terminal_success).any().item()):
        raise ValueError("valid_mask cells are allowed only on terminal-success trajectories")

    terminal_success_count = int(terminal_success.sum().item())
    metrics = {
        "dominance/terminal_success_trajectory_count": float(terminal_success_count),
        "dominance/terminal_success_trajectory_rate": _rate(
            terminal_success_count, batch_size
        ),
    }
    for column, horizon in enumerate(horizons):
        active_mask = terminal_success & valid_mask[:, column]
        active_count = int(active_mask.sum().item())
        active_mean = (
            float(probe_values[active_mask, column].mean().item())
            if active_count
            else 0.0
        )
        metrics[f"dominance/readiness_active_mean_h{horizon}"] = active_mean
        metrics[f"dominance/readiness_active_count_h{horizon}"] = float(active_count)
        metrics[f"dominance/readiness_active_valid_rate_h{horizon}"] = _rate(
            active_count, terminal_success_count
        )
    return metrics


@torch.no_grad()
def compute_readiness_dominance(
    probe_values: torch.Tensor,
    valid_mask: torch.Tensor,
    terminal_success: torch.Tensor,
    positive_trajectory_mask: torch.Tensor,
    group_ids: Sequence[object],
    *,
    n: int,
    strict_branch_margin: int = 1,
    min_common_positions: int = 2,
) -> tuple[DominanceResult, dict[str, float]]:
    """Compute direct readiness dominance on each pair's active common support."""
    if probe_values.ndim != 2 or probe_values.shape[1] == 0:
        raise ValueError("probe_values must have shape (batch, positions>0)")
    if not probe_values.is_floating_point():
        raise ValueError("probe_values must have floating dtype")
    if valid_mask.shape != probe_values.shape:
        raise ValueError("valid_mask and probe_values must have the same shape")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must have boolean dtype")
    batch_size = probe_values.shape[0]
    _require_bool_vector("terminal_success", terminal_success, batch_size)
    _require_bool_vector("positive_trajectory_mask", positive_trajectory_mask, batch_size)
    if (
        valid_mask.device != probe_values.device
        or terminal_success.device != probe_values.device
        or positive_trajectory_mask.device != probe_values.device
    ):
        raise ValueError("all dominance tensors must be on the same device")
    if len(group_ids) != batch_size:
        raise ValueError("group_ids length must match probe_values batch size")
    if not bool(torch.isfinite(probe_values).all().item()):
        raise ValueError("probe_values must be finite")
    if not bool(((probe_values >= 0.0) & (probe_values <= 1.0)).all().item()):
        raise ValueError("probe_values must be in [0, 1]")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if (
        not isinstance(strict_branch_margin, int)
        or isinstance(strict_branch_margin, bool)
        or strict_branch_margin <= 0
    ):
        raise ValueError("strict_branch_margin must be a positive integer")
    if (
        not isinstance(min_common_positions, int)
        or isinstance(min_common_positions, bool)
        or min_common_positions <= 0
    ):
        raise ValueError("min_common_positions must be a positive integer")

    eligible_mask = terminal_success & positive_trajectory_mask
    dominance_matrix = torch.zeros(
        (batch_size, batch_size), dtype=torch.bool, device=probe_values.device
    )
    groups = _group_indices(group_ids)
    comparable_pairs_by_group = [0] * len(groups)
    same_group_eligible_pairs = 0
    comparable_pairs = 0
    crossing_pairs = 0
    common_position_counts: list[int] = []
    strict_delta = strict_branch_margin / n

    for group_index, indices in enumerate(groups):
        for left_offset, left in enumerate(indices):
            if not bool(eligible_mask[left].item()):
                continue
            for right in indices[left_offset + 1 :]:
                if not bool(eligible_mask[right].item()):
                    continue
                same_group_eligible_pairs += 1
                common = valid_mask[left] & valid_mask[right]
                common_count = int(common.sum().item())
                if common_count < min_common_positions:
                    continue
                comparable_pairs += 1
                comparable_pairs_by_group[group_index] += 1
                common_position_counts.append(common_count)
                delta = probe_values[left, common] - probe_values[right, common]
                left_dominates = bool(
                    torch.all(delta >= 0).item() and torch.any(delta >= strict_delta).item()
                )
                right_dominates = bool(
                    torch.all(delta <= 0).item() and torch.any(-delta >= strict_delta).item()
                )
                dominance_matrix[left, right] = left_dominates
                dominance_matrix[right, left] = right_dominates
                if bool(torch.any(delta > 0).item() and torch.any(delta < 0).item()):
                    crossing_pairs += 1

    dominated_mask = dominance_matrix.any(dim=0)
    frontier_mask = eligible_mask & ~dominated_mask
    group_has_dominance = torch.zeros(batch_size, dtype=torch.bool, device=probe_values.device)
    eligible_group_count = 0
    dominance_group_count = 0
    frontier_sizes: list[int] = []
    for group_index, indices in enumerate(groups):
        if comparable_pairs_by_group[group_index] > 0:
            eligible_group_count += 1
            frontier_sizes.append(int(frontier_mask[indices].sum().item()))
        group_edge_count = int(dominance_matrix[indices][:, indices].sum().item())
        if group_edge_count > 0:
            dominance_group_count += 1
            group_has_dominance[indices] = True

    eligible_count = int(eligible_mask.sum().item())
    dominated_count = int(dominated_mask.sum().item())
    frontier_count = int(frontier_mask.sum().item())
    if common_position_counts:
        common_mean = float(sum(common_position_counts) / len(common_position_counts))
        common_min = float(min(common_position_counts))
        common_max = float(max(common_position_counts))
    else:
        common_mean = common_min = common_max = 0.0
    metrics = {
        "dominance/eligible_group_rate": _rate(eligible_group_count, len(groups)),
        "dominance/group_with_dominance_rate": _rate(dominance_group_count, len(groups)),
        "dominance/eligible_positive_rate": _rate(eligible_count, batch_size),
        "dominance/dominated_positive_rate": _rate(dominated_count, eligible_count),
        "dominance/frontier_fraction": _rate(frontier_count, eligible_count),
        "dominance/frontier_size_mean": (
            float(sum(frontier_sizes) / len(frontier_sizes)) if frontier_sizes else 0.0
        ),
        "dominance/profile_cross_rate": _rate(crossing_pairs, comparable_pairs),
        "dominance/same_group_eligible_pair_count": float(same_group_eligible_pairs),
        "dominance/comparable_pair_count": float(comparable_pairs),
        "dominance/pair_coverage_rate": _rate(comparable_pairs, same_group_eligible_pairs),
        "dominance/common_positions_mean": common_mean,
        "dominance/common_positions_min": common_min,
        "dominance/common_positions_max": common_max,
    }
    return (
        DominanceResult(
            dominance_matrix=dominance_matrix,
            eligible_mask=eligible_mask,
            frontier_mask=frontier_mask,
            dominated_mask=dominated_mask,
            group_has_dominance=group_has_dominance,
        ),
        metrics,
    )


def _validate_dominance_result(
    dominance: DominanceResult,
    group_ids: Sequence[object],
    batch_size: int,
    device: torch.device,
) -> list[list[int]]:
    matrix = dominance.dominance_matrix
    if matrix.shape != (batch_size, batch_size) or matrix.dtype is not torch.bool:
        raise ValueError("dominance_matrix must be a boolean (batch, batch) tensor")
    vector_names = (
        "eligible_mask",
        "frontier_mask",
        "dominated_mask",
        "group_has_dominance",
    )
    for name in vector_names:
        tensor = getattr(dominance, name)
        if tensor.shape != (batch_size,) or tensor.dtype is not torch.bool:
            raise ValueError(f"{name} must be a boolean (batch,) tensor")
        if tensor.device != device:
            raise ValueError("all dominance tensors must be on the advantages device")
    if matrix.device != device:
        raise ValueError("all dominance tensors must be on the advantages device")
    if bool(torch.diagonal(matrix).any().item()):
        raise ValueError("dominance_matrix must not contain self-dominance")
    if not torch.equal(matrix.any(dim=0), dominance.dominated_mask):
        raise ValueError("dominated_mask must match direct incoming dominance edges")
    expected_frontier = dominance.eligible_mask & ~dominance.dominated_mask
    if not torch.equal(expected_frontier, dominance.frontier_mask):
        raise ValueError("frontier_mask must be the directly nondominated eligible set")
    if bool((dominance.dominated_mask & ~dominance.eligible_mask).any().item()):
        raise ValueError("only eligible trajectories may be dominated")

    groups = _group_indices(group_ids)
    group_index_by_row = [-1] * batch_size
    for group_index, indices in enumerate(groups):
        for index in indices:
            group_index_by_row[index] = group_index
        expected_has_edge = bool(matrix[indices][:, indices].any().item())
        if any(
            bool(dominance.group_has_dominance[index].item()) != expected_has_edge
            for index in indices
        ):
            raise ValueError("group_has_dominance must match direct within-group edges")
    for source, target in matrix.nonzero(as_tuple=False).tolist():
        if group_index_by_row[source] != group_index_by_row[target]:
            raise ValueError("dominance_matrix must not contain cross-group edges")
    return groups


@torch.no_grad()
def apply_frontier_reweighting(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[object],
    dominance: DominanceResult,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Apply token-mean-mass-preserving trajectory weights to direct dominance groups."""
    if advantages.ndim != 2:
        raise ValueError("advantages must have shape (batch, response_tokens)")
    if not advantages.is_floating_point():
        raise ValueError("advantages must have floating dtype")
    if response_mask.shape != advantages.shape:
        raise ValueError("response_mask and advantages must have the same shape")
    if response_mask.device != advantages.device:
        raise ValueError("response_mask and advantages must be on the same device")
    if len(group_ids) != advantages.shape[0]:
        raise ValueError("group_ids length must match advantages batch size")
    if response_mask.is_floating_point() and not bool(torch.isfinite(response_mask).all().item()):
        raise ValueError("response_mask must be finite and binary")
    if not bool(((response_mask == 0) | (response_mask == 1)).all().item()):
        raise ValueError("response_mask must be binary")
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("advantages must be finite")

    mask = response_mask.bool()
    for row in range(advantages.shape[0]):
        active = advantages[row, mask[row]]
        padding = advantages[row, ~mask[row]]
        if active.numel() and not torch.equal(active, active[0].expand_as(active)):
            raise ValueError(
                "standard GRPO advantages must be constant within each trajectory"
            )
        if padding.numel() and not torch.equal(padding, torch.zeros_like(padding)):
            raise ValueError("standard GRPO padding advantages must be exactly zero")

    groups = _validate_dominance_result(
        dominance, group_ids, advantages.shape[0], advantages.device
    )
    positive_mass_by_row = (
        advantages.clamp_min(0) * mask.to(dtype=advantages.dtype)
    ).sum(dim=-1)
    weights = torch.ones(
        advantages.shape[0], dtype=advantages.dtype, device=advantages.device
    )
    scales: list[torch.Tensor] = []
    group_masses: list[tuple[list[int], torch.Tensor]] = []
    total_before = advantages.new_zeros(())
    for indices in groups:
        if not bool(dominance.group_has_dominance[indices[0]].item()):
            continue
        eligible_indices = [
            index for index in indices if bool(dominance.eligible_mask[index].item())
        ]
        frontier_indices = [
            index for index in indices if bool(dominance.frontier_mask[index].item())
        ]
        dominated_indices = [
            index for index in indices if bool(dominance.dominated_mask[index].item())
        ]
        before = positive_mass_by_row[eligible_indices].sum()
        frontier_mass = positive_mass_by_row[frontier_indices].sum()
        if not bool(torch.isfinite(before).item()) or before <= 0:
            raise ValueError("eligible positive advantage mass must be finite and positive")
        if not bool(torch.isfinite(frontier_mass).item()) or frontier_mass <= 0:
            raise ValueError("frontier mass must be finite and positive")
        scale = before / frontier_mass
        if not bool(torch.isfinite(scale).item()):
            raise ValueError("frontier scale must be finite")
        weights[dominated_indices] = 0
        weights[frontier_indices] = scale
        scales.append(scale)
        group_masses.append((eligible_indices, before))
        total_before += before

    new_advantages = advantages * weights.unsqueeze(-1)
    new_positive_mass_by_row = (
        new_advantages.clamp_min(0) * mask.to(dtype=advantages.dtype)
    ).sum(dim=-1)
    total_after = advantages.new_zeros(())
    residuals: list[torch.Tensor] = []
    for eligible_indices, before in group_masses:
        after = new_positive_mass_by_row[eligible_indices].sum()
        residual = (after - before).abs()
        tolerance = max(1.0e-6, 1.0e-6 * abs(float(before.item())))
        if not bool(torch.isfinite(after).item()) or not bool(torch.isfinite(residual).item()):
            raise ValueError("reweighted positive advantage mass must be finite")
        if float(residual.item()) > tolerance:
            raise ValueError("frontier reweighting violates positive mass conservation")
        total_after += after
        residuals.append(residual)

    if scales:
        scale_tensor = torch.stack(scales).float()
        scale_mean = float(scale_tensor.mean().item())
        scale_max = float(scale_tensor.max().item())
        scale_p50 = float(torch.quantile(scale_tensor, 0.50).item())
        scale_p90 = float(torch.quantile(scale_tensor, 0.90).item())
        scale_p99 = float(torch.quantile(scale_tensor, 0.99).item())
    else:
        scale_mean = scale_max = scale_p50 = scale_p90 = scale_p99 = 0.0
    residual_max = (
        float(torch.stack(residuals).max().item()) if residuals else 0.0
    )
    metrics = {
        "dominance/positive_mass_before": float(total_before.item()),
        "dominance/positive_mass_after": float(total_after.item()),
        "dominance/mass_residual_max": residual_max,
        "dominance/frontier_scale_mean": scale_mean,
        "dominance/frontier_scale_max": scale_max,
        "dominance/frontier_scale_p50": scale_p50,
        "dominance/frontier_scale_p90": scale_p90,
        "dominance/frontier_scale_p99": scale_p99,
        "dominance/skipped_invalid_mass_groups": 0.0,
    }
    return new_advantages, weights, metrics
