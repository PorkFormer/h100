"""Strict extraction helpers for the existing synchronous reward pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import ListConfig


@dataclass(frozen=True)
class NormalizedRewardOutput:
    reward_tensor: torch.Tensor
    extra_info: dict[str, Any]


def verifier_pipeline_fingerprint(
    *,
    reward_manager_name: str = "naive",
    reward_manager_source: str = "register",
    reward_manager_module_path: str | None = None,
    reward_manager_module_name: str | None = None,
    custom_reward_function_path: str | None = None,
    custom_reward_function_name: str = "compute_score",
    custom_reward_kwargs: Any = None,
    reward_kwargs: Any = None,
    sandbox_fusion: Any = None,
) -> str:
    """Hash all local code and configuration that can change verifier ``acc``."""

    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("verifier fingerprint settings must be finite")
            return value
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, Mapping) or hasattr(value, "items"):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple, ListConfig)):
            return [normalize(item) for item in value]
        raise ValueError(
            f"verifier fingerprint setting {type(value).__name__} is not JSON-compatible"
        )

    verl_root = Path(__file__).resolve().parents[2]
    source_roots = (
        verl_root / "experimental" / "reward_loop" / "reward_manager",
        verl_root / "utils" / "reward_score",
    )
    sources = sorted(
        path
        for root in source_roots
        for path in root.rglob("*.py")
        if path.is_file()
    )
    sources.extend(
        (
            verl_root / "trainer" / "ppo" / "reward.py",
            verl_root / "experimental" / "reward_loop" / "reward_loop.py",
        )
    )
    if custom_reward_function_path:
        custom_path = Path(custom_reward_function_path).expanduser().resolve()
        if not custom_path.is_file():
            raise ValueError("custom reward function path must be a local file")
        sources.append(custom_path)
    if reward_manager_source not in ("register", "importlib"):
        raise ValueError("reward manager source must be register or importlib")
    if reward_manager_source == "importlib":
        if not reward_manager_module_path:
            raise ValueError("importlib reward manager requires a local module path")
        module_path = Path(reward_manager_module_path).expanduser().resolve()
        if not module_path.is_file():
            raise ValueError("reward manager module path must be a local file")
        sources.append(module_path)
    digest = hashlib.sha256()
    digest.update(b"obcf-verifier-pipeline-v1\0")
    settings = json.dumps(
        {
            "reward_manager_name": reward_manager_name,
            "reward_manager_source": reward_manager_source,
            "reward_manager_module_name": (
                reward_manager_module_name if reward_manager_source == "importlib" else None
            ),
            "custom_reward_function_name": custom_reward_function_name,
            "custom_reward_kwargs": normalize(
                {} if custom_reward_kwargs is None else custom_reward_kwargs
            ),
            "reward_kwargs": normalize({} if reward_kwargs is None else reward_kwargs),
            "sandbox_fusion": normalize({} if sandbox_fusion is None else sandbox_fusion),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest.update(len(settings).to_bytes(8, "big"))
    digest.update(settings)
    for path in sorted(set(sources), key=lambda item: str(item.resolve())):
        try:
            identity = path.relative_to(verl_root).as_posix()
        except ValueError:
            identity = f"custom/{path.name}"
        encoded_identity = identity.encode()
        digest.update(len(encoded_identity).to_bytes(8, "big"))
        digest.update(encoded_identity)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def extract_binary_accuracy(
    reward_output: NormalizedRewardOutput,
    *,
    expected_count: int,
    metric_name: str = "acc",
) -> torch.Tensor:
    """Extract exact binary verifier accuracy without shaped-reward fallback."""
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 0:
        raise ValueError("expected_count must be a nonnegative integer")
    if metric_name not in reward_output.extra_info:
        raise ValueError(f"reward output is missing required verifier metric {metric_name!r}")
    value = reward_output.extra_info[metric_name]
    try:
        accuracy = torch.as_tensor(value, device=reward_output.reward_tensor.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"verifier metric {metric_name!r} must be numeric") from error
    if accuracy.ndim != 1 or accuracy.numel() != expected_count:
        raise ValueError(f"verifier metric {metric_name!r} must have exactly {expected_count} rows")
    if accuracy.dtype.is_complex or not bool(torch.isfinite(accuracy).all().item()):
        raise ValueError(f"verifier metric {metric_name!r} must be finite and real")
    if not bool(((accuracy == 0) | (accuracy == 1)).all().item()):
        raise ValueError(f"verifier metric {metric_name!r} must be binary")
    return accuracy.to(dtype=torch.float32)
