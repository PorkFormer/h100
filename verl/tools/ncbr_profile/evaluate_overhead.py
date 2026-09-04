#!/usr/bin/env python3
"""Evaluate diagnostics equivalence and workload-normalized time/memory gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def overhead_fraction(off: float, on: float) -> float | str:
    if off <= 0:
        return "unavailable"
    return (on - off) / off


def evaluate(fixed_replay: dict[str, Any] | list[dict[str, Any]], stage_costs: dict[str, Any]) -> dict[str, Any]:
    replay_records = fixed_replay if isinstance(fixed_replay, list) else [fixed_replay]
    if not replay_records:
        raise ValueError("at least one fixed replay receipt is required")
    stage_overheads = {
        name: overhead_fraction(float(values["off"]), float(values["on"])) for name, values in stage_costs.items()
    }
    fixed_time_values = [record.get("actor_time_overhead_fraction", "unavailable") for record in replay_records]
    fixed_time = (
        max(float(value) for value in fixed_time_values)
        if all(isinstance(value, int | float) for value in fixed_time_values)
        else "unavailable"
    )
    time_values = [value for value in (fixed_time, *stage_overheads.values()) if isinstance(value, int | float)]
    memory_values = [
        record.get(name, "unavailable")
        for record in replay_records
        for name in ("peak_allocated_overhead_fraction", "peak_reserved_overhead_fraction")
    ]
    numeric_memory = [value for value in memory_values if isinstance(value, int | float)]
    max_time = max(time_values) if time_values else "unavailable"
    max_memory = max(numeric_memory) if len(numeric_memory) == len(memory_values) else "unavailable"
    equivalence_pass = all(
        bool(record.get("equivalence_pass", False)) and record.get("status", "PASS") == "PASS"
        for record in replay_records
    )
    result = {
        "schema_version": "ncbr-diagnostics-overhead-gate-v1",
        "fixed_replay_receipt_count": len(replay_records),
        "equivalence_pass": equivalence_pass,
        "fixed_replay_actor_time_overhead_fraction": fixed_time,
        "stage_unit_cost_overhead_fractions": stage_overheads,
        "max_time_overhead_fraction": max_time,
        "fixed_replay_memory_overhead_fractions": memory_values,
        "max_memory_overhead_fraction": max_memory,
        "time_gate_pass": isinstance(max_time, int | float) and max_time <= 0.03,
        "memory_gate_pass": isinstance(max_memory, int | float) and max_memory <= 0.02,
    }
    result["status"] = (
        "PASS" if equivalence_pass and result["time_gate_pass"] and result["memory_gate_pass"] else "FAIL"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-replay", type=Path, action="append", required=True)
    parser.add_argument("--stage-costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.fixed_replay],
        json.loads(args.stage_costs.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit("diagnostics overhead/equivalence gate failed")


if __name__ == "__main__":
    main()
