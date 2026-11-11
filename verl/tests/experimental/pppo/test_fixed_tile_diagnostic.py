"""Gradient check for the experimental Gate 2 execution adapter, not acceptance."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('bias', [False, True])
def test_fixed_tile_autograd_gradcheck(bias):
    path = Path(__file__).resolve().parents[4] / 'experiments/pppo_ncbr_matched/stable_linear_diagnostic.py'
    spec = importlib.util.spec_from_file_location('stable_linear_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    w = torch.randn(7, 5, dtype=torch.double, requires_grad=True)
    b = torch.randn(7, dtype=torch.double, requires_grad=True) if bias else None
    assert torch.autograd.gradcheck(lambda *args: module.FixedTileLinear.apply(*args, 4), (x, w, b))


def test_inplace_temperature_scaling_after_custom_forward():
    import copy
    path = Path(__file__).resolve().parents[4] / 'experiments/pppo_ncbr_matched/stable_linear_diagnostic.py'
    spec = importlib.util.spec_from_file_location('stable_linear_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(42)
    reference = torch.nn.Linear(5, 7).double()
    actual = copy.deepcopy(reference)
    module.install_fixed_tile_linear(actual, 4)
    x = torch.randn(1, 6, 5, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = reference(x).squeeze(0), actual(y).squeeze(0)
    a.div_(1.3)
    b.div_(1.3)
    a.square().sum().backward()
    b.square().sum().backward()
    for left, right in [(a, b), (x.grad, y.grad), (reference.weight.grad, actual.weight.grad)]:
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
