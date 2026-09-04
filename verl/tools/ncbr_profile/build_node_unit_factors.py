#!/usr/bin/env python3
"""Map calibration/crossover throughput factors onto profile unit costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(calibration: dict, crossover: dict | None = None) -> dict:
    components = calibration["components"]

    def factors(name: str) -> dict[str, float]:
        values = components[name]["node_throughput_factor"]
        return {node: float(values[node]) for node in ("A", "B")}

    normal = factors("normal_decode_tokens_per_second")
    reward = factors("reward_full_response_tokens_per_second")
    actor = factors("actor_valid_tokens_per_second")
    candidate = factors("candidate_batches_per_second")
    if calibration.get("crossover_required"):
        if crossover is None or crossover.get("status") != "PASS":
            raise ValueError("component crossover receipt is required after a >5% node calibration gap")
        required = {
            "request_control",
            "continuation_prefill",
            "tail_decode",
            "long_reward_row",
            "long_reward_token",
            "normal_decode",
            "actor",
            "candidate",
        }
        if set(crossover.get("node_throughput_factor", {})) != required:
            raise ValueError("component crossover factors are incomplete")
        mapping = {
            "u_request": "request_control",
            "u_cont_input": "continuation_prefill",
            "u_tail_decode": "tail_decode",
            "u_long_row": "long_reward_row",
            "u_long_token": "long_reward_token",
            "u_normal": "normal_decode",
            "u_actor": "actor",
            "u_candidate": "candidate",
        }
        result = {
            node: {
                unit: float(crossover["node_throughput_factor"][component][node])
                for unit, component in mapping.items()
            }
            for node in ("A", "B")
        }
        method = "fixed_component_crossover"
    else:
        result = {
            node: {
                "u_request": normal[node],
                "u_cont_input": normal[node],
                "u_tail_decode": normal[node],
                "u_long_row": reward[node],
                "u_long_token": reward[node],
                "u_normal": normal[node],
                "u_actor": actor[node],
                "u_candidate": candidate[node],
            }
            for node in ("A", "B")
        }
        method = "baseline_calibration_below_5_percent"
    if any(value <= 0 for node in result.values() for value in node.values()):
        raise ValueError("node unit-cost factors must all be positive")
    return {
        "schema_version": "qwen3-1p7b-node-unit-cost-factors-v1",
        "status": "PASS",
        "method": method,
        "node_unit_cost_factors": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--crossover", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    crossover = json.loads(args.crossover.read_text(encoding="utf-8")) if args.crossover else None
    result = build(calibration, crossover)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
