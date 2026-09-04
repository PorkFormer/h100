#!/usr/bin/env python3
"""Build the required cumulative S300 curve axes from per-step metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build(records: list[dict], arm: str) -> list[dict]:
    totals = {
        "candidate_prompts": 0.0,
        "normal_decode_tokens": 0.0,
        "continuation_input_tokens": 0.0,
        "continuation_tail_decode_tokens": 0.0,
        "actor_valid_tokens": 0.0,
        "wall_clock_seconds": 0.0,
        "gpu_hours": 0.0,
    }
    output = []

    def global_step(item: dict) -> int:
        return int(item.get("step_metrics", item)["training/global_step"])

    for record in sorted(records, key=global_step):
        metrics = record.get("step_metrics", record)
        timing = record.get("trainer_timing_raw", {})
        step = int(metrics["training/global_step"])
        values = {
            "candidate_prompts": float(metrics["train/generated_prompt_groups"]),
            "normal_decode_tokens": float(metrics["train/generated_response_tokens"]),
            "continuation_input_tokens": float(metrics.get("boundary_return/continuation_input_tokens", 0.0)),
            "continuation_tail_decode_tokens": float(metrics.get("boundary_return/tail_decode_tokens", 0.0)),
            "actor_valid_tokens": float(metrics["actor_diagnostics/all/token_count"]),
            "wall_clock_seconds": float(timing.get("step", metrics.get("timing_s/step"))),
        }
        if arm == "baseline" and (
            values["continuation_input_tokens"] != 0 or values["continuation_tail_decode_tokens"] != 0
        ):
            raise ValueError("Baseline continuation coordinates must remain exactly zero")
        values["gpu_hours"] = values["wall_clock_seconds"] * 8 / 3600
        for name, value in values.items():
            totals[name] += value
        output.append({"optimizer_step": step, **totals})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "v1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = build(records, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["optimizer_step"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
