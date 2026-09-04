from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.workers import engine_workers


class _FakeDiagnostics:
    def __init__(self, config):
        self.enabled = bool(config.get("enable", False))

    def accumulate(self, model_output, data):
        del model_output, data

    def finalize(self, group):
        del group
        return SimpleNamespace(metrics={}, reduction_calls=int(self.enabled))


class _FakeEngine:
    def __init__(self):
        self.module = torch.nn.Linear(2, 1, bias=False)
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.01)
        self.scaler = None
        self.train_batch_calls = 0

    def train_batch(self, data, loss_function):
        self.train_batch_calls += 1
        prediction = self.module(data["features"])
        loss, metrics = loss_function(model_output={"prediction": prediction}, data=data, dp_group=None)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"model_output": prediction.detach(), "metrics": metrics}

    def get_data_parallel_group(self):
        return None


def test_real_actor_replay_restores_model_optimizer_and_rng(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_workers, "ActorDiagnosticsAccumulator", _FakeDiagnostics)
    worker = engine_workers.TrainingWorker.__new__(engine_workers.TrainingWorker)
    worker.engine = _FakeEngine()

    def loss_fn(*, model_output, data, dp_group=None):
        del data, dp_group
        loss = model_output["prediction"].square().mean()
        return loss, {"policy_loss": loss.detach()}

    worker.loss_fn = loss_fn
    batch = TensorDict({"features": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, batch_size=2)
    _, diagnostics, _ = worker._fixed_actor_replay(
        batch,
        {"enable": True},
        {"repeats": 2, "receipt_dir": str(tmp_path)},
    )
    receipt = json.loads((tmp_path / "rank_00000.json").read_text(encoding="utf-8"))
    assert worker.engine.train_batch_calls == 6
    assert receipt["unmeasured_off_and_on_diagnostics_warmup"] is True
    assert receipt["balanced_measurement_order"] is True
    assert receipt["measurement_order"] == [False, True, True, False]
    assert receipt["unmeasured_dp_barrier_before_each_observation"] is True
    assert len(receipt["observations"]) == 4
    assert receipt["equivalence_pass"]
    assert receipt["max_diagnostic_reduction_calls_per_optimizer_step"] == 1
    assert diagnostics.reduction_calls == 1
