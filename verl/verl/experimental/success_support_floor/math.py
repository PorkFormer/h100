"""Pure numerical primitives for the Budgeted Success-Support Floor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SupportFloorResult:
    current_seq_logprob: torch.Tensor
    log_ratio: torch.Tensor
    shortfall: torch.Tensor
    active_fraction: torch.Tensor


@dataclass(frozen=True)
class DualState:
    lambda_value: float
    violation_ema: float


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def compute_support_floor(
    current_logprob: torch.Tensor,
    reference_seq_logprob: torch.Tensor,
    *,
    alpha: float,
    response_mask: torch.Tensor | None = None,
) -> SupportFloorResult:
    """Compute witness-wise log ratios and one-sided shortfalls in FP32."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if current_logprob.numel() == 0:
        raise ValueError("support batch must be nonempty")

    current = current_logprob.float()
    if response_mask is not None:
        if current.ndim != 2 or response_mask.shape != current.shape:
            raise ValueError("response_mask must match a two-dimensional token log-prob tensor")
        current = (current * response_mask.to(dtype=torch.float32)).sum(dim=-1)
    elif current.ndim != 1:
        raise ValueError("current_logprob must be a sequence vector when response_mask is absent")

    reference = reference_seq_logprob.float()
    if reference.ndim != 1 or reference.shape != current.shape:
        raise ValueError("reference_seq_logprob must match the support sequence batch")
    _require_finite("current log probability", current)
    _require_finite("reference log probability", reference)

    log_ratio = current - reference
    shortfall = torch.relu(math.log(alpha) - log_ratio)
    _require_finite("support shortfall", shortfall)
    return SupportFloorResult(
        current_seq_logprob=current,
        log_ratio=log_ratio,
        shortfall=shortfall,
        active_fraction=(shortfall > 0).float().mean(),
    )


def update_dual_state(
    *,
    lambda_value: float,
    violation_ema: float,
    observed_constraint: float,
    delta: float,
    dual_lr: float,
    ema_beta: float,
    lambda_max: float,
) -> DualState:
    """Apply one projected stochastic dual-ascent update."""
    values = (lambda_value, violation_ema, observed_constraint, delta, dual_lr, lambda_max)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dual state and observations must be finite")
    if lambda_value < 0.0 or delta < 0.0 or dual_lr < 0.0 or lambda_max < lambda_value:
        raise ValueError("invalid nonnegative dual-update parameters")
    if not 0.0 <= ema_beta < 1.0:
        raise ValueError("ema_beta must be in [0, 1)")
    next_ema = ema_beta * violation_ema + (1.0 - ema_beta) * observed_constraint
    next_lambda = min(lambda_max, max(0.0, lambda_value + dual_lr * (next_ema - delta)))
    return DualState(lambda_value=next_lambda, violation_ema=next_ema)
