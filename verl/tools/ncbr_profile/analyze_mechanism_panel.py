#!/usr/bin/env python3
"""Derive continuation and long-reward unit costs from a frozen panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.natural_continuation_boundary_return.profiling import (  # noqa: E402
    ProfileInterval,
    interval_union_seconds,
    normalized_unit_cost,
)


def analyze(receipt: dict) -> dict:
    intervals = [ProfileInterval(**item) for item in receipt["profiling_intervals"]]
    by_name: dict[str, list[ProfileInterval]] = {}
    for interval in intervals:
        by_name.setdefault(interval.name, []).append(interval)
    request_count = float(receipt["request_count"])
    workloads = {
        "request_count": request_count,
        "continuation_input_tokens": float(receipt["continuation_input_tokens"]),
        "tail_decode_tokens": float(receipt["tail_decode_tokens"]),
        "long_reward_rows": float(receipt["long_reward_rows"]),
        "long_reward_full_response_tokens": float(receipt["long_reward_full_response_tokens"]),
    }
    continuation = interval_union_seconds(by_name.get("boundary_continuation", []))
    engine = interval_union_seconds(
        [
            interval
            for name in ("continuation_queue", "continuation_prefill_engine", "continuation_decode_engine")
            for interval in by_name.get(name, [])
        ]
    )
    seconds = {
        "u_request": max(continuation - engine, 0.0),
        "u_cont_input": interval_union_seconds(by_name.get("continuation_prefill_engine", [])),
        "u_tail_decode": interval_union_seconds(by_name.get("continuation_decode_engine", [])),
        "u_long_row": interval_union_seconds(by_name.get("long_reward_batch_build", [])),
        "u_long_token": interval_union_seconds(by_name.get("long_reward_model_forward", [])),
    }
    denominators = {
        "u_request": "request_count",
        "u_cont_input": "continuation_input_tokens",
        "u_tail_decode": "tail_decode_tokens",
        "u_long_row": "long_reward_rows",
        "u_long_token": "long_reward_full_response_tokens",
    }
    units = {
        name: normalized_unit_cost(value, workloads[denominators[name]]) for name, value in seconds.items()
    }
    status = "PASS" if all(isinstance(value, int | float) for value in units.values()) else "FAIL"
    return {
        "schema_version": "qwen3-1p7b-ncbr-mechanism-panel-analysis-v1",
        "status": status,
        "workloads": workloads,
        "stage_seconds": seconds,
        "unit_costs": units,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit("mechanism panel unit costs are unavailable")


if __name__ == "__main__":
    main()
