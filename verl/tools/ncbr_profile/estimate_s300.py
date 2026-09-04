#!/usr/bin/env python3
"""Estimate S300 wall time, GPU-hours, and disk from measured stage costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCENARIOS = ("early", "moderate", "stress")
ARMS = ("baseline", "v1")


def estimate(measurements: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for scenario in SCENARIOS:
        arms = {}
        for arm in ARMS:
            record = measurements["arms"][arm]
            training = 300 * float(record["training_step_seconds"][scenario])
            step0_validation = float(record["validation_step0_seconds"])
            periodic_validation = 30 * float(record["validation_periodic_seconds"])
            checkpoints = 6 * float(record["checkpoint_seconds"])
            wall = training + step0_validation + periodic_validation + checkpoints
            uncertainty_fraction = float(record["uncertainty_fraction"][scenario])
            arms[arm] = {
                "training_seconds": training,
                "step0_validation_seconds": step0_validation,
                "periodic_validation_seconds": periodic_validation,
                "checkpoint_seconds": checkpoints,
                "wall_clock_seconds": wall,
                "wall_clock_uncertainty_seconds": wall * uncertainty_fraction,
                "gpu_hours": wall * 8 / 3600,
                "checkpoint_disk_bytes": 6 * int(record["checkpoint_bytes"]),
            }
        results[scenario] = {
            "phase_label": "early_phase_lower_bound" if scenario == "early" else scenario,
            "arms": arms,
            "parallel_wall_clock_seconds": max(item["wall_clock_seconds"] for item in arms.values()),
            "sequential_wall_clock_seconds": sum(item["wall_clock_seconds"] for item in arms.values()),
            "combined_gpu_hours": sum(item["gpu_hours"] for item in arms.values()),
            "combined_checkpoint_disk_bytes": sum(item["checkpoint_disk_bytes"] for item in arms.values()),
        }
    return {
        "schema_version": "qwen3-1p7b-ncbr-s300-estimate-v1",
        "status": "PASS",
        "optimizer_steps": 300,
        "periodic_validation_count": 30,
        "checkpoint_steps": [50, 100, 150, 200, 250, 300],
        "scenarios": results,
        "assumptions": measurements.get("assumptions", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
