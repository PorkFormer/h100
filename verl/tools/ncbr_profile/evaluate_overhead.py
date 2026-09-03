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


def evaluate(fixed_replay: dict[str, Any], stage_costs: dict[str, Any]) -> dict[str, Any]:
    stage_overheads = {
        name: overhead_fraction(float(values["off"]), float(values["on"])) for name, values in stage_costs.items()
    }
    fixed_time = fixed_replay.get("actor_time_overhead_fraction", "unavailable")
    time_values = [value for value in (fixed_time, *stage_overheads.values()) if isinstance(value, int | float)]
    memory_values = [
        fixed_replay.get("peak_allocated_overhead_fraction", "unavailable"),
        fixed_replay.get("peak_reserved_overhead_fraction", "unavailable"),
    ]
    numeric_memory = [value for value in memory_values if isinstance(value, int | float)]
    max_time = max(time_values) if time_values else "unavailable"
    max_memory = max(numeric_memory) if numeric_memory else "unavailable"
    equivalence_pass = bool(fixed_replay.get("equivalence_pass", False))
    result = {
        "schema_version": "ncbr-diagnostics-overhead-gate-v1",
        "equivalence_pass": equivalence_pass,
        "fixed_replay_actor_time_overhead_fraction": fixed_time,
        "stage_unit_cost_overhead_fractions": stage_overheads,
        "max_time_overhead_fraction": max_time,
        "peak_allocated_overhead_fraction": fixed_replay.get("peak_allocated_overhead_fraction", "unavailable"),
        "peak_reserved_overhead_fraction": fixed_replay.get("peak_reserved_overhead_fraction", "unavailable"),
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
    parser.add_argument("--fixed-replay", type=Path, required=True)
    parser.add_argument("--stage-costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.fixed_replay.read_text(encoding="utf-8")),
        json.loads(args.stage_costs.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit("diagnostics overhead/equivalence gate failed")


if __name__ == "__main__":
    main()
