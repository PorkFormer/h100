#!/usr/bin/env python3
"""Aggregate all distributed ranks of a real fixed actor replay."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def aggregate(receipt_dir: Path, world_size: int) -> dict[str, Any]:
    paths = sorted(receipt_dir.glob("rank_*.json"))
    if len(paths) != world_size:
        raise ValueError(f"fixed actor replay rank count mismatch: expected {world_size}, got {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ranks = [int(record["rank"]) for record in records]
    if ranks != list(range(world_size)) or any(int(record["world_size"]) != world_size for record in records):
        raise ValueError(f"fixed actor replay rank/world identity mismatch: {ranks}")
    equivalence = all(bool(record.get("equivalence_pass")) for record in records)
    reductions = max(int(record["max_diagnostic_reduction_calls_per_optimizer_step"]) for record in records)

    def worst(name: str) -> float | str:
        values = [record.get(name, "unavailable") for record in records]
        numeric = [float(value) for value in values if isinstance(value, int | float)]
        return max(numeric) if len(numeric) == len(values) else "unavailable"

    result = {
        "schema_version": "real-fixed-actor-batch-replay-v1",
        "world_size": world_size,
        "rank_receipts": [str(path.resolve()) for path in paths],
        "equivalence_pass": equivalence,
        "max_diagnostic_reduction_calls_per_optimizer_step": reductions,
        "actor_time_overhead_fraction": worst("actor_time_overhead_fraction"),
        "median_rank_actor_time_overhead_fraction": statistics.median(
            float(record["actor_time_overhead_fraction"]) for record in records
        ),
        "peak_allocated_overhead_fraction": worst("peak_allocated_overhead_fraction"),
        "peak_reserved_overhead_fraction": worst("peak_reserved_overhead_fraction"),
    }
    result["status"] = "PASS" if equivalence and reductions <= 1 else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.receipt_dir, args.world_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit("distributed fixed actor replay failed")


if __name__ == "__main__":
    main()
