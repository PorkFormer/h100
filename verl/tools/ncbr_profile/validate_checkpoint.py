#!/usr/bin/env python3
"""Validate and hash a complete resumable actor checkpoint without mutating it."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(value, dict):
        raise SystemExit(f"checkpoint file is not a mapping: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    if root.name != f"global_step_{args.step}":
        raise SystemExit(f"checkpoint directory does not match Step {args.step}: {root}")
    tracker = root.parent / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(args.step):
        raise SystemExit("checkpoint tracker does not identify the requested step")
    actor = root / "actor"
    shard_sets = {
        "model": sorted(actor.glob(f"model_world_size_{args.world_size}_rank_*.pt")),
        "optimizer": sorted(actor.glob(f"optim_world_size_{args.world_size}_rank_*.pt")),
        "extra": sorted(actor.glob(f"extra_state_world_size_{args.world_size}_rank_*.pt")),
    }
    for name, paths in shard_sets.items():
        if len(paths) != args.world_size:
            raise SystemExit(f"{name} shard count mismatch: expected {args.world_size}, got {len(paths)}")
    data_path = root / "data.pt"
    data_state = load_mapping(data_path)
    if not {"_snapshot", "_steps_since_snapshot", "_iterator_finished"}.issubset(data_state):
        raise SystemExit("dataloader state is incomplete")
    del data_state
    for path in shard_sets["extra"]:
        state = load_mapping(path)
        if not {"lr_scheduler", "rng"}.issubset(state):
            raise SystemExit(f"scheduler/RNG state is incomplete: {path}")
        del state
    for kind in ("model", "optimizer"):
        for path in shard_sets[kind]:
            state = load_mapping(path)
            if not state:
                raise SystemExit(f"empty {kind} shard: {path}")
            del state
            gc.collect()
    hf_root = actor / "huggingface"
    required_hf = [hf_root / "config.json", hf_root / "model.safetensors", hf_root / "tokenizer.json"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_hf):
        raise SystemExit("Hugging Face export is incomplete")
    all_files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    result = {
        "schema_version": "ncbr-complete-checkpoint-v1",
        "status": "PASS",
        "step": args.step,
        "world_size": args.world_size,
        "checkpoint": str(root),
        "counts": {name: len(paths) for name, paths in shard_sets.items()},
        "file_count": len(all_files),
        "files": {str(path.relative_to(root)): sha256(path) for path in all_files},
        "resume_state": {
            "model": True,
            "optimizer": True,
            "rng": True,
            "scheduler": True,
            "dataloader": True,
            "step_counter": True,
            "hf_model": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
