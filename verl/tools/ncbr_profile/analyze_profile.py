#!/usr/bin/env python3
"""Analyze timer-DAG JSONL without inventing a fixed-cost residual."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.natural_continuation_boundary_return.profiling import (  # noqa: E402
    ProfileInterval,
    analyze_interval_dag,
    coefficient_of_variation,
    interval_union_seconds,
    normalized_unit_cost,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    workloads = {}
    for item in args.workload:
        name, raw = item.split("=", 1)
        workloads[name] = float(raw)
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    analyses = []
    step_walls = []
    for record in records:
        intervals = [ProfileInterval(**item) for item in record.get("intervals", [])]
        analysis = analyze_interval_dag(intervals)
        by_name = {}
        for interval in intervals:
            by_name.setdefault(interval.name, []).append(interval)
        requested_unit_costs = {
            name: normalized_unit_cost(interval_union_seconds(by_name.get(name, [])), workload)
            for name, workload in workloads.items()
        }
        metrics = record.get("step_metrics", {})
        automatic_workloads = {
            "request_count": float(metrics.get("boundary_return/continuation_request_count", 0.0)),
            "continuation_input_tokens": float(metrics.get("boundary_return/continuation_input_tokens", 0.0)),
            "tail_decode_tokens": float(metrics.get("boundary_return/tail_decode_tokens", 0.0)),
            "long_reward_rows": float(metrics.get("boundary_return/long_reward_rows", 0.0)),
            "long_reward_full_response_tokens": float(
                metrics.get("boundary_return/long_reward_full_response_tokens", 0.0)
            ),
            "normal_decode_tokens": float(metrics.get("train/generated_response_tokens", 0.0)),
            "normal_trajectories": float(metrics.get("train/generated_trajectories", 0.0)),
            "actor_valid_tokens": float(metrics.get("actor_diagnostics/all/token_count", 0.0)),
            "candidate_batches": float(metrics.get("train/num_gen_batches", 0.0)),
        }
        if automatic_workloads["actor_valid_tokens"] <= 0:
            actor_intervals = by_name.get("actor_update", [])
            actor_token_values = [
                float(interval.metadata.get("actor_valid_tokens", 0.0)) for interval in actor_intervals
            ]
            automatic_workloads["actor_valid_tokens"] = max(actor_token_values, default=0.0)
        stage_seconds = {
            "u_request": float(metrics.get("boundary_return/continuation_control_exclusive_seconds", 0.0)),
            "u_cont_input": interval_union_seconds(by_name.get("continuation_prefill_engine", [])),
            "u_tail_decode": interval_union_seconds(by_name.get("continuation_decode_engine", [])),
            "u_long_row": interval_union_seconds(by_name.get("long_reward_batch_build", [])),
            "u_long_token": interval_union_seconds(by_name.get("long_reward_model_forward", [])),
            "u_normal": interval_union_seconds(by_name.get("normal_rollout", [])),
            "u_actor": interval_union_seconds(
                [
                    interval
                    for name in ("old_log_prob", "reference_log_prob", "advantage", "actor_update")
                    for interval in by_name.get(name, [])
                ]
            ),
            "u_candidate": interval_union_seconds(
                [interval for name in ("short_reward", "dynamic_sampling_filter") for interval in by_name.get(name, [])]
            ),
        }
        denominator = {
            "u_request": automatic_workloads["request_count"],
            "u_cont_input": automatic_workloads["continuation_input_tokens"],
            "u_tail_decode": automatic_workloads["tail_decode_tokens"],
            "u_long_row": automatic_workloads["long_reward_rows"],
            "u_long_token": automatic_workloads["long_reward_full_response_tokens"],
            "u_normal": automatic_workloads["normal_decode_tokens"],
            "u_actor": automatic_workloads["actor_valid_tokens"],
            "u_candidate": automatic_workloads["candidate_batches"],
        }
        automatic_unit_costs = {
            name: normalized_unit_cost(seconds, denominator[name]) for name, seconds in stage_seconds.items()
        }
        timing = record.get("trainer_timing_raw", {})
        if "step" in timing:
            step_walls.append(float(timing["step"]))
        covered_wall = interval_union_seconds(intervals)
        other_wall = max(float(timing["step"]) - covered_wall, 0.0) if "step" in timing else "unavailable"
        analyses.append(
            {
                **analysis,
                "workloads": automatic_workloads,
                "stage_seconds": stage_seconds,
                "unit_costs": automatic_unit_costs,
                "requested_unit_costs": requested_unit_costs,
                "covered_interval_union_seconds": covered_wall,
                "other_wall_seconds": other_wall,
            }
        )
    stable_window = step_walls[1:4] if len(step_walls) < 6 else step_walls[1:6]
    cv = coefficient_of_variation(stable_window)
    stable_records = analyses[1:4] if len(analyses) < 6 else analyses[1:6]
    aggregate_unit_costs = {}
    for name in (
        "u_request",
        "u_cont_input",
        "u_tail_decode",
        "u_long_row",
        "u_long_token",
        "u_normal",
        "u_actor",
        "u_candidate",
    ):
        values = [record["unit_costs"][name] for record in stable_records]
        numeric = [float(value) for value in values if isinstance(value, int | float)]
        aggregate_unit_costs[name] = statistics.median(numeric) if numeric else "unavailable"
    natural_requests = sum(record["workloads"]["request_count"] for record in stable_records)
    output = {
        "schema_version": "ncbr-profile-analysis-v1",
        "record_count": len(records),
        "records": analyses,
        "step_wall_cv": cv,
        "extension_required": isinstance(cv, float) and cv > 0.10 and len(step_walls) < 6,
        "unstable": isinstance(cv, float) and cv > 0.10 and len(step_walls) >= 6,
        "u_fixed": "unavailable",
        "other_wall_method": "interval_union",
        "stable_window_unit_cost_medians": aggregate_unit_costs,
        "natural_request_count_stable_window": natural_requests,
        "mechanism_coverage_insufficient": natural_requests < 20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
