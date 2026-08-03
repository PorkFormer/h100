"""Atomic checkpoint state contract for BSSF primal-dual training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from verl.trainer.config import SuccessSupportFloorConfig

STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SuccessSupportFloorState:
    global_step: int
    lambda_value: float
    violation_ema: float
    support_update_count: int
    last_support_step: int
    cache_fingerprint: str
    config_fingerprint: str


def scientific_config_fingerprint(config: SuccessSupportFloorConfig) -> str:
    """Fingerprint all scientific/sampling settings while allowing cache relocation."""
    payload = asdict(config)
    payload.pop("cache_path", None)
    payload.pop("_target_", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_state(path: str | Path, state: SuccessSupportFloorState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": STATE_SCHEMA_VERSION, **asdict(state)}
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
) -> SuccessSupportFloorState:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"BSSF checkpoint state is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported BSSF state schema_version {payload.get('schema_version')}")
    state = SuccessSupportFloorState(
        global_step=int(payload["global_step"]),
        lambda_value=float(payload["lambda_value"]),
        violation_ema=float(payload["violation_ema"]),
        support_update_count=int(payload["support_update_count"]),
        last_support_step=int(payload["last_support_step"]),
        cache_fingerprint=str(payload["cache_fingerprint"]),
        config_fingerprint=str(payload["config_fingerprint"]),
    )
    if state.global_step != expected_global_step:
        raise ValueError(
            f"BSSF state global_step mismatch: {state.global_step} != {expected_global_step}"
        )
    if state.cache_fingerprint != expected_cache_fingerprint:
        raise ValueError("BSSF state cache_fingerprint mismatch")
    if state.config_fingerprint != expected_config_fingerprint:
        raise ValueError("BSSF state config_fingerprint mismatch")
    if not math.isfinite(state.lambda_value) or not 0.0 <= state.lambda_value <= lambda_max:
        raise ValueError("BSSF state lambda is non-finite or outside configured bounds")
    if not math.isfinite(state.violation_ema):
        raise ValueError("BSSF state violation_ema must be finite")
    if state.support_update_count < 0:
        raise ValueError("BSSF state support_update_count must be nonnegative")
    if state.last_support_step < -1 or state.last_support_step > state.global_step:
        raise ValueError("BSSF state last_support_step is invalid")
    return state
