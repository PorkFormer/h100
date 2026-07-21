"""Pure PyTorch mathematics for on-policy Probe credit redistribution."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import torch

from verl.utils import as_torch_index, group_mean_std


@torch.no_grad()
def compute_probe_pseudo_rewards(probe_values: torch.Tensor) -> torch.Tensor:
    """Compute local recoverability changes between adjacent Probe positions."""
    if probe_values.ndim != 2 or probe_values.shape[1] < 2:
        raise ValueError("probe_values must have shape (batch, positions>=2)")
    return probe_values[:, 1:] - probe_values[:, :-1]


@torch.no_grad()
def compute_probe_temporal_returns(pseudo_rewards: torch.Tensor, rho: float) -> torch.Tensor:
    """Discount local Probe progress backward across temporal segments."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if pseudo_rewards.ndim != 2:
        raise ValueError("pseudo_rewards must have shape (batch, segments)")
    returns = torch.zeros_like(pseudo_rewards)
    running = torch.zeros(pseudo_rewards.shape[0], dtype=pseudo_rewards.dtype, device=pseudo_rewards.device)
    for segment in range(pseudo_rewards.shape[1] - 1, -1, -1):
        running = pseudo_rewards[:, segment] + rho * running
        returns[:, segment] = running
    return returns


@torch.no_grad()
def compute_group_relative_probe_advantages(
    temporal_returns: torch.Tensor,
    group_ids: Sequence[Any],
    *,
    norm_by_std: bool = True,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalize each Probe segment within prompt groups using GRPO's std convention."""
    if temporal_returns.ndim != 2:
        raise ValueError("temporal_returns must have shape (batch, segments)")
    if len(group_ids) != temporal_returns.shape[0]:
        raise ValueError("group_ids length must match temporal_returns batch size")
    group_index = as_torch_index(group_ids, device=temporal_returns.device)
    advantages = torch.zeros_like(temporal_returns)
    degenerate = torch.zeros_like(temporal_returns, dtype=torch.bool)
    for segment in range(temporal_returns.shape[1]):
        scores = temporal_returns[:, segment]
        means, stds, counts = group_mean_std(scores, group_index, eps=0.0, device=scores.device)
        segment_degenerate = (counts[group_index] <= 1) | (stds[group_index] <= epsilon)
        centered = scores - means[group_index]
        normalized = centered / (stds[group_index] + epsilon) if norm_by_std else centered
        advantages[:, segment] = torch.where(segment_degenerate, 0.0, normalized)
        degenerate[:, segment] = segment_degenerate
    metrics = {
        "probe_credit/degenerate_rate": float(degenerate.float().mean().item()) if degenerate.numel() else 0.0,
        "probe_credit/temporal_return_mean": float(temporal_returns.mean().item())
        if temporal_returns.numel()
        else 0.0,
    }
    return advantages, metrics


@torch.no_grad()
def build_probe_token_correction(
    segment_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    relative_boundaries: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.9),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Map four Probe segments to response tokens and center their covered mass."""
    if segment_advantages.ndim != 2 or segment_advantages.shape[1] != 4:
        raise ValueError("segment_advantages must have shape (batch, 4)")
    if response_mask.ndim != 2 or response_mask.shape[0] != segment_advantages.shape[0]:
        raise ValueError("response_mask must have shape (batch, response_tokens)")
    if tuple(relative_boundaries) != (0.0, 0.25, 0.5, 0.75, 0.9):
        raise ValueError("the first version requires canonical Probe token boundaries")

    correction = torch.zeros_like(response_mask, dtype=segment_advantages.dtype)
    for row in range(response_mask.shape[0]):
        length = int(response_mask[row].sum().item())
        boundaries = (0, length // 4, length // 2, (3 * length) // 4, math_floor_90(length), length)
        covered_length = boundaries[4]
        if covered_length == 0:
            continue
        weighted_sum = segment_advantages.new_zeros(())
        for segment in range(4):
            segment_length = boundaries[segment + 1] - boundaries[segment]
            weighted_sum += segment_length * segment_advantages[row, segment]
        weighted_mean = weighted_sum / covered_length
        for segment in range(4):
            start, end = boundaries[segment], boundaries[segment + 1]
            correction[row, start:end] = segment_advantages[row, segment] - weighted_mean
    correction *= response_mask.to(dtype=correction.dtype)
    residual = (correction * response_mask).sum(dim=-1).abs()
    metrics = {
        "probe_credit/correction_abs_mean": float(correction.abs().mean().item()) if correction.numel() else 0.0,
        "probe_credit/correction_abs_max": float(correction.abs().max().item()) if correction.numel() else 0.0,
        "probe_credit/zero_mass_residual_max": float(residual.max().item()) if residual.numel() else 0.0,
    }
    return correction, metrics


def math_floor_90(length: int) -> int:
    """Compute floor(0.90 * length) without binary floating-point boundary drift."""
    return (9 * length) // 10


@torch.no_grad()
def apply_probe_credit_redistribution(
    data: Any,
    correction: torch.Tensor,
    *,
    coef: float,
    enable: bool,
) -> dict[str, float]:
    """Add Probe correction to standard GRPO advantages without touching rewards."""
    tensor_batch: MutableMapping[str, torch.Tensor] = data.batch if hasattr(data, "batch") else data
    if not enable:
        return {"probe_credit/enabled": 0.0}
    terminal_advantages = tensor_batch["advantages"].clone()
    if correction.shape != terminal_advantages.shape:
        raise ValueError("Probe correction shape must match terminal advantages")
    final_advantages = terminal_advantages + float(coef) * correction
    tensor_batch["terminal_advantages"] = terminal_advantages
    tensor_batch["advantages"] = final_advantages
    tensor_batch["returns"] = final_advantages
    return {
        "probe_credit/enabled": 1.0,
        "probe_credit/terminal_advantage_mean": _mean(terminal_advantages),
        "probe_credit/terminal_advantage_std": _std(terminal_advantages),
        "probe_credit/correction_mean": _mean(correction),
        "probe_credit/correction_std": _std(correction),
        "probe_credit/final_advantage_mean": _mean(final_advantages),
        "probe_credit/final_advantage_std": _std(final_advantages),
    }


def _mean(tensor: torch.Tensor) -> float:
    return float(tensor.mean().item()) if tensor.numel() else 0.0


def _std(tensor: torch.Tensor) -> float:
    return float(tensor.float().std(unbiased=False).item()) if tensor.numel() else 0.0
