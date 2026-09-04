#!/usr/bin/env python3
"""Build component node factors from fixed P0 crossover measurements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(calibration_a: dict, calibration_b: dict, panel_a: dict, panel_b: dict) -> dict:
    raw = {
        "normal_decode": {
            "A": float(calibration_a["normal_decode_tokens_per_second"]),
            "B": float(calibration_b["normal_decode_tokens_per_second"]),
        },
        "actor": {
            "A": float(calibration_a["actor_valid_tokens_per_second"]),
            "B": float(calibration_b["actor_valid_tokens_per_second"]),
        },
        "candidate": {
            "A": float(calibration_a["candidate_batches_per_second"]),
            "B": float(calibration_b["candidate_batches_per_second"]),
        },
    }
    unit_mapping = {
        "request_control": "u_request",
        "continuation_prefill": "u_cont_input",
        "tail_decode": "u_tail_decode",
        "long_reward_row": "u_long_row",
        "long_reward_token": "u_long_token",
    }
    for component, unit in unit_mapping.items():
        values = {"A": float(panel_a["unit_costs"][unit]), "B": float(panel_b["unit_costs"][unit])}
        if min(values.values()) <= 0:
            raise ValueError(f"crossover unit cost must be positive: {component}: {values}")
        raw[component] = {node: 1.0 / value for node, value in values.items()}
    factors = {}
    for component, values in raw.items():
        if min(values.values()) <= 0:
            raise ValueError(f"crossover throughput must be positive: {component}: {values}")
        reference = math.sqrt(values["A"] * values["B"])
        factors[component] = {node: value / reference for node, value in values.items()}
    return {
        "schema_version": "qwen3-1p7b-component-crossover-v1",
        "status": "PASS",
        "raw_throughput": raw,
        "node_throughput_factor": factors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-a", type=Path, required=True)
    parser.add_argument("--calibration-b", type=Path, required=True)
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.calibration_a.read_text(encoding="utf-8")),
        json.loads(args.calibration_b.read_text(encoding="utf-8")),
        json.loads(args.panel_a.read_text(encoding="utf-8")),
        json.loads(args.panel_b.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
