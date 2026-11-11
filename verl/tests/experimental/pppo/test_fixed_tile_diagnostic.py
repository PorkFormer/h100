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
