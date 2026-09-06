#!/usr/bin/env python3
"""Fail-closed recursive diff for fully resolved OmegaConf job configs."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_ALLOWED_PATHS = (
    "actor_rollout_ref.rollout.boundary_return.mode",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "trainer.dynamic_sampling_gate_receipt_path",
    "trainer.profile_interval_path",
    "ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR",
    "ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_ID",
    "ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME",
    "ray_kwargs.ray_init.runtime_env.env_vars.XDG_CONFIG_HOME",
    "ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_WORKSPACE_BASE",
    "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPYCACHEPREFIX",
    "ray_kwargs.ray_init.runtime_env.env_vars.TORCH_EXTENSIONS_DIR",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resolved(path: Path) -> Any:
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


def recursive_diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        records: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                records.append({"path": child, "baseline": "<missing>", "v1": right[key]})
            elif key not in right:
                records.append({"path": child, "baseline": left[key], "v1": "<missing>"})
            else:
                records.extend(recursive_diff(left[key], right[key], child))
        return records
    if isinstance(left, list) and isinstance(right, list):
        records = []
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left):
                records.append({"path": child, "baseline": "<missing>", "v1": right[index]})
            elif index >= len(right):
                records.append({"path": child, "baseline": left[index], "v1": "<missing>"})
            else:
                records.extend(recursive_diff(left[index], right[index], child))
        return records
    if type(left) is not type(right) or left != right:
        return [{"path": path, "baseline": left, "v1": right}]
    return []


def build_receipt(baseline_path: Path, v1_path: Path, allowed: tuple[str, ...]) -> dict[str, Any]:
    differences = recursive_diff(load_resolved(baseline_path), load_resolved(v1_path))
    for item in differences:
        item["allowed"] = any(fnmatch.fnmatchcase(item["path"], pattern) for pattern in allowed)
    unexpected = [item for item in differences if not item["allowed"]]
    mode_path = "actor_rollout_ref.rollout.boundary_return.mode"
    mode = next((item for item in differences if item["path"] == mode_path), None)
    mode_valid = mode is not None and mode["baseline"] == "off" and mode["v1"] == "replace"
    if not mode_valid:
        unexpected.append(
            {
                "path": mode_path,
                "baseline": None if mode is None else mode["baseline"],
                "v1": None if mode is None else mode["v1"],
                "allowed": False,
                "reason": "required off -> replace semantic delta is absent or invalid",
            }
        )
    return {
        "schema_version": "ncbr-resolved-config-diff-v1",
        "baseline": {"path": str(baseline_path.resolve()), "sha256": sha256(baseline_path)},
        "v1": {"path": str(v1_path.resolve()), "sha256": sha256(v1_path)},
        "allowed_paths": list(allowed),
        "differences": differences,
        "unexpected_differences": unexpected,
        "semantic_delta": {"path": mode_path, "baseline": "off", "v1": "replace"},
        "status": "PASS" if not unexpected else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow", action="append", default=[])
    args = parser.parse_args()
    allowed = tuple(args.allow) if args.allow else DEFAULT_ALLOWED_PATHS
    try:
        receipt = build_receipt(args.baseline, args.v1, allowed)
    except Exception as exc:
        receipt = {
            "schema_version": "ncbr-resolved-config-diff-v1",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
