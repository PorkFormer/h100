"""Atomic checkpoint state for OBCF primal-dual training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

from verl.trainer.config import OnPolicyBudgetedCapabilityFloorConfig

STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OnPolicyBudgetedCapabilityFloorState:
    global_step: int
    lambda_value: float
    violation_ema: float
    ema_initialized: bool
    constraint_observation_count: int
    last_constraint_step: int
    cache_fingerprint: str
    config_fingerprint: str


def scientific_config_fingerprint(config: OnPolicyBudgetedCapabilityFloorConfig) -> str:
    payload = asdict(config)
    payload.pop("cache_path", None)
    payload.pop("_target_", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_state(path: str | Path, state: OnPolicyBudgetedCapabilityFloorState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = asdict(state)
    fields["lambda"] = fields.pop("lambda_value")
    payload = {"schema_version": STATE_SCHEMA_VERSION, **fields}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_state(
    path: str | Path,
    *,
    expected_global_step: int,
    expected_cache_fingerprint: str,
    expected_config_fingerprint: str,
    lambda_max: float,
) -> OnPolicyBudgetedCapabilityFloorState:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"OBCF checkpoint state is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "global_step",
        "lambda",
        "violation_ema",
        "ema_initialized",
        "constraint_observation_count",
        "last_constraint_step",
        "cache_fingerprint",
        "config_fingerprint",
    }
    if set(payload) != required:
        raise ValueError("OBCF checkpoint state fields are malformed")
    if payload["schema_version"] != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported OBCF state schema_version")
    if not isinstance(payload["ema_initialized"], bool):
        raise ValueError("OBCF state ema_initialized must be boolean")
    for name in ("global_step", "constraint_observation_count", "last_constraint_step"):
        if not isinstance(payload[name], int) or isinstance(payload[name], bool):
            raise ValueError(f"OBCF state {name} must be an integer")
    for name in ("lambda", "violation_ema"):
        if not isinstance(payload[name], Real) or isinstance(payload[name], bool):
            raise ValueError(f"OBCF state {name} must be a real number")
    state = OnPolicyBudgetedCapabilityFloorState(
        global_step=int(payload["global_step"]),
        lambda_value=float(payload["lambda"]),
        violation_ema=float(payload["violation_ema"]),
        ema_initialized=payload["ema_initialized"],
        constraint_observation_count=int(payload["constraint_observation_count"]),
        last_constraint_step=int(payload["last_constraint_step"]),
        cache_fingerprint=str(payload["cache_fingerprint"]),
        config_fingerprint=str(payload["config_fingerprint"]),
    )
    if state.global_step != expected_global_step:
        raise ValueError("OBCF state global_step mismatch")
    if state.cache_fingerprint != expected_cache_fingerprint:
        raise ValueError("OBCF state cache_fingerprint mismatch")
    if state.config_fingerprint != expected_config_fingerprint:
        raise ValueError("OBCF state config_fingerprint mismatch")
    if not math.isfinite(state.lambda_value) or not 0.0 <= state.lambda_value <= lambda_max:
        raise ValueError("OBCF state lambda is non-finite or outside configured bounds")
    if not math.isfinite(state.violation_ema) or not 0.0 <= state.violation_ema <= 1.0:
        raise ValueError("OBCF state violation_ema must be finite and in [0, 1]")
    if state.global_step < 0:
        raise ValueError("OBCF state global_step must be nonnegative")
    if state.ema_initialized != (state.constraint_observation_count > 0):
        raise ValueError("OBCF EMA initialization disagrees with observation count")
    if not 0 <= state.constraint_observation_count <= state.global_step:
        raise ValueError("OBCF constraint_observation_count is outside valid step bounds")
    if state.last_constraint_step < -1 or state.last_constraint_step > state.global_step:
        raise ValueError("OBCF last_constraint_step is invalid")
    if (state.constraint_observation_count == 0) != (state.last_constraint_step == -1):
        raise ValueError("OBCF last_constraint_step disagrees with observation count")
    if state.constraint_observation_count == 0 and state.violation_ema != 0.0:
        raise ValueError("uninitialized OBCF state must have zero violation_ema")
    return state
