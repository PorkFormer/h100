#!/usr/bin/env python3
"""Compare identical Baseline/P0 calibration workloads across the two nodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

THROUGHPUT_KEYS = (
    "normal_decode_tokens_per_second",
    "reward_full_response_tokens_per_second",
    "actor_valid_tokens_per_second",
    "candidate_batches_per_second",
)


def compare(left: dict[str, Any], right: dict[str, Any], threshold: float = 0.05) -> dict[str, Any]:
    comparisons = {}
    crossover_required = False
    for key in THROUGHPUT_KEYS:
        values = {"A": float(left[key]), "B": float(right[key])}
        if min(values.values()) <= 0:
            raise ValueError(f"calibration throughput must be positive: {key}: {values}")
        geometric_mean = math.sqrt(values["A"] * values["B"])
        relative_gap = abs(values["A"] - values["B"]) / geometric_mean
        if relative_gap > threshold:
            crossover_required = True
        comparisons[key] = {
            "raw": values,
            "geometric_mean_reference": geometric_mean,
            "relative_gap": relative_gap,
            "node_throughput_factor": {node: value / geometric_mean for node, value in values.items()},
        }
    gpu_values = {"A": float(left["gpu_utilization_mean"]), "B": float(right["gpu_utilization_mean"])}
    return {
        "schema_version": "qwen3-1p7b-cross-node-calibration-v1",
        "threshold": threshold,
        "components": comparisons,
        "gpu_utilization_mean": gpu_values,
        "crossover_required": crossover_required,
        "status": "PASS_REQUIRES_CROSSOVER" if crossover_required else "PASS",
        "normalization_rule": "normalized_cost_equals_raw_cost_times_node_throughput_factor",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-a", type=Path, required=True)
    parser.add_argument("--node-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    result = compare(
        json.loads(args.node_a.read_text(encoding="utf-8")),
        json.loads(args.node_b.read_text(encoding="utf-8")),
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
