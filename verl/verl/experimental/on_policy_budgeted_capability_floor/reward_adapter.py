"""Strict extraction helpers for the existing synchronous reward pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizedRewardOutput:
    reward_tensor: torch.Tensor
    extra_info: dict[str, Any]


def extract_binary_accuracy(
    reward_output: NormalizedRewardOutput,
    *,
    expected_count: int,
    metric_name: str = "acc",
) -> torch.Tensor:
    """Extract exact binary verifier accuracy without shaped-reward fallback."""
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 0:
        raise ValueError("expected_count must be a nonnegative integer")
    if metric_name not in reward_output.extra_info:
        raise ValueError(f"reward output is missing required verifier metric {metric_name!r}")
    value = reward_output.extra_info[metric_name]
    try:
        accuracy = torch.as_tensor(value, device=reward_output.reward_tensor.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"verifier metric {metric_name!r} must be numeric") from error
    if accuracy.ndim != 1 or accuracy.numel() != expected_count:
        raise ValueError(f"verifier metric {metric_name!r} must have exactly {expected_count} rows")
    if accuracy.dtype.is_complex or not bool(torch.isfinite(accuracy).all().item()):
        raise ValueError(f"verifier metric {metric_name!r} must be finite and real")
    if not bool(((accuracy == 0) | (accuracy == 1)).all().item()):
        raise ValueError(f"verifier metric {metric_name!r} must be binary")
    return accuracy.to(dtype=torch.float32)
