"""Projected scalar dual updates shared by capability constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectedDualState:
    lambda_value: float
    violation_ema: float
    ema_initialized: bool


def update_projected_dual(
    *,
    lambda_value: float,
    violation_ema: float,
    ema_initialized: bool,
    observed_constraint: float,
    delta: float,
    dual_lr: float,
    ema_beta: float,
    lambda_max: float,
) -> ProjectedDualState:
    """Update an EMA residual and project scalar dual ascent onto its bounds."""
    values = (
        lambda_value,
        violation_ema,
        observed_constraint,
        delta,
        dual_lr,
        ema_beta,
        lambda_max,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dual state and observations must be finite")
    if not isinstance(ema_initialized, bool):
        raise ValueError("ema_initialized must be a bool")
    if not 0.0 <= lambda_value <= lambda_max:
        raise ValueError("lambda_value must be within [0, lambda_max]")
    if delta < 0.0 or dual_lr < 0.0 or lambda_max < 0.0:
        raise ValueError("dual update parameters must be nonnegative")
    if not 0.0 <= ema_beta < 1.0:
        raise ValueError("ema_beta must be in [0, 1)")

    next_ema = (
        ema_beta * violation_ema + (1.0 - ema_beta) * observed_constraint
        if ema_initialized
        else observed_constraint
    )
    next_lambda = min(
        lambda_max,
        max(0.0, lambda_value + dual_lr * (next_ema - delta)),
    )
    return ProjectedDualState(
        lambda_value=next_lambda,
        violation_ema=next_ema,
        ema_initialized=True,
    )
