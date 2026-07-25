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
