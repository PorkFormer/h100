from __future__ import annotations

import torch

from verl.experimental.natural_continuation_boundary_return.overhead import FixedBatchReplayHarness


def test_fixed_batch_harness_restores_full_state_and_proves_diagnostics_inert(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    targets = torch.tensor([[0.5]])
    calls = []

    def step(diagnostics_enabled: bool):
        calls.append(diagnostics_enabled)
        output = model(inputs)
        loss = torch.square(output - targets).mean()
        if diagnostics_enabled:
            _ = (output.detach().double().sum(), output.detach().double().square().sum())
        loss.backward()
        optimizer.step()
        return loss

    result = FixedBatchReplayHarness(model, optimizer, step).run(repeats=2)
    assert calls == [False, True, False, True, False, True]
    assert result["unmeasured_off_and_on_diagnostics_warmup"] is True
    assert len(result["observations"]) == 4
    assert result["equivalence_pass"]
    assert all(result["equivalence"].values())
    assert result["peak_allocated_overhead_fraction"] == "unavailable"
    assert result["peak_reserved_overhead_fraction"] == "unavailable"
