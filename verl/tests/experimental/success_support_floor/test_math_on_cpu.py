import math

import pytest
import torch

from verl.experimental.success_support_floor.math import (
    compute_support_floor,
    update_dual_state,
)


def test_base_and_above_floor_have_zero_shortfall():
    reference = torch.tensor([-10.0, -20.0])
    current = reference + torch.tensor([0.0, math.log(0.75)])
    result = compute_support_floor(current, reference, alpha=0.5)

    assert torch.equal(result.shortfall, torch.zeros(2))
    assert result.active_fraction.item() == 0.0


def test_below_floor_shortfall_and_gradient_increase_log_likelihood():
    current = torch.tensor([-12.0], requires_grad=True)
    result = compute_support_floor(current, torch.tensor([-10.0]), alpha=0.5)

    result.shortfall.mean().backward()

    assert result.shortfall.item() == pytest.approx(2.0 + math.log(0.5))
    assert current.grad.item() == pytest.approx(-1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_accumulate_as_float32(dtype):
    token_log_probs = torch.tensor([[-1.0, -2.0, -3.0]], dtype=dtype)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    result = compute_support_floor(
        token_log_probs,
        torch.tensor([-2.0]),
        alpha=0.5,
        response_mask=mask,
    )

    assert result.current_seq_logprob.dtype == torch.float32
    assert result.current_seq_logprob.item() == pytest.approx(-3.0)


def test_non_finite_fails_closed():
    with pytest.raises(ValueError, match="finite"):
        compute_support_floor(torch.tensor([float("nan")]), torch.tensor([0.0]), alpha=0.5)


def test_empty_batch_fails_closed():
    with pytest.raises(ValueError, match="nonempty"):
        compute_support_floor(torch.empty(0), torch.empty(0), alpha=0.5)


def test_dual_update_uses_ema_residual_and_clamps():
    state = update_dual_state(
        lambda_value=0.2,
        violation_ema=0.0,
        observed_constraint=0.3,
        delta=0.05,
        dual_lr=1.0,
        ema_beta=0.5,
        lambda_max=0.3,
    )

    assert state.violation_ema == pytest.approx(0.15)
    assert state.lambda_value == pytest.approx(0.3)
