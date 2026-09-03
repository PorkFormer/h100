# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same-process fixed-batch diagnostics overhead/equivalence harness."""

from __future__ import annotations

import copy
import hashlib
import io
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


def _fingerprint(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@dataclass(frozen=True)
class ReplayObservation:
    diagnostics_enabled: bool
    loss: float
    loss_fingerprint: str
    gradient_fingerprint: str
    parameter_fingerprint: str
    optimizer_fingerprint: str
    rng_fingerprint: str
    elapsed_seconds: float
    peak_allocated_bytes: int | str
    peak_reserved_bytes: int | str


class FixedBatchReplayHarness:
    """Restore model, optimizer and RNG before every alternating off/on replay."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step_fn: Callable[[bool], Any]):
        self.model = model
        self.optimizer = optimizer
        self.step_fn = step_fn
        self._model_state = copy.deepcopy(model.state_dict())
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())
        self._python_rng = random.getstate()
        self._numpy_rng = copy.deepcopy(np.random.get_state())
        self._torch_rng = torch.random.get_rng_state().clone()
        self._cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def _restore(self) -> None:
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(copy.deepcopy(self._optimizer_state))
        self.optimizer.zero_grad(set_to_none=True)
        random.setstate(self._python_rng)
        np.random.set_state(self._numpy_rng)
        torch.random.set_rng_state(self._torch_rng)
        if self._cuda_rng is not None:
            torch.cuda.set_rng_state_all(self._cuda_rng)

    def _run_one(self, enabled: bool) -> ReplayObservation:
        self._restore()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = self.step_fn(enabled)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
            peak_allocated: int | str = torch.cuda.max_memory_allocated()
            peak_reserved: int | str = torch.cuda.max_memory_reserved()
        else:
            started = time.perf_counter()
            loss = self.step_fn(enabled)
            elapsed = time.perf_counter() - started
            peak_allocated = "unavailable"
            peak_reserved = "unavailable"
        loss_tensor = torch.as_tensor(loss).detach().cpu()
        gradients = {
            name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
            for name, parameter in self.model.named_parameters()
        }
        return ReplayObservation(
            diagnostics_enabled=enabled,
            loss=float(loss_tensor.item()),
            loss_fingerprint=_fingerprint(loss_tensor),
            gradient_fingerprint=_fingerprint(gradients),
            parameter_fingerprint=_fingerprint(self.model.state_dict()),
            optimizer_fingerprint=_fingerprint(self.optimizer.state_dict()),
            rng_fingerprint=_fingerprint(
                {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.random.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
            ),
            elapsed_seconds=elapsed,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )

    def run(self, repeats: int = 2) -> dict[str, Any]:
        if repeats <= 0:
            raise ValueError("fixed replay repeats must be positive")
        observations = [self._run_one(enabled) for _ in range(repeats) for enabled in (False, True)]
        off = [item for item in observations if not item.diagnostics_enabled]
        on = [item for item in observations if item.diagnostics_enabled]
        equivalence_fields = (
            "loss_fingerprint",
            "gradient_fingerprint",
            "parameter_fingerprint",
            "optimizer_fingerprint",
            "rng_fingerprint",
        )
        equivalence = {field: len({getattr(item, field) for item in observations}) == 1 for field in equivalence_fields}
        off_time = statistics.median(item.elapsed_seconds for item in off)
        on_time = statistics.median(item.elapsed_seconds for item in on)

        def peak_delta(field: str) -> float | str:
            off_values = [getattr(item, field) for item in off]
            on_values = [getattr(item, field) for item in on]
            if any(value == "unavailable" for value in (*off_values, *on_values)):
                return "unavailable"
            off_peak = max(int(value) for value in off_values)
            on_peak = max(int(value) for value in on_values)
            return (on_peak - off_peak) / off_peak if off_peak else 0.0

        return {
            "schema_version": "fixed-actor-batch-replay-v1",
            "observations": [item.__dict__ for item in observations],
            "equivalence": equivalence,
            "equivalence_pass": all(equivalence.values()),
            "actor_time_overhead_fraction": (on_time - off_time) / off_time if off_time else "unavailable",
            "peak_allocated_overhead_fraction": peak_delta("peak_allocated_bytes"),
            "peak_reserved_overhead_fraction": peak_delta("peak_reserved_bytes"),
        }
