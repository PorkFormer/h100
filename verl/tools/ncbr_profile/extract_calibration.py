#!/usr/bin/env python3
"""Derive one-node calibration throughput from measured intervals/workloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.natural_continuation_boundary_return.profiling import (
    ProfileInterval,
    interval_union_seconds,
)


def extract(profile: dict, gpu_samples: dict) -> dict:
    intervals = [ProfileInterval(**item) for item in profile["intervals"]]
    by_name: dict[str, list[ProfileInterval]] = {}
    for interval in intervals:
        by_name.setdefault(interval.name, []).append(interval)
    metrics = profile["step_metrics"]
    normal_tokens = float(metrics["train/generated_response_tokens"])
    actor_tokens = float(metrics["actor_diagnostics/all/token_count"])
    candidate_batches = float(metrics["train/num_gen_batches"])
    seconds = {
        "normal": interval_union_seconds(by_name.get("normal_rollout", [])),
        "reward": interval_union_seconds(by_name.get("short_reward", [])),
        "actor": interval_union_seconds(
            [
                interval
                for name in ("old_log_prob", "reference_log_prob", "advantage", "actor_update")
                for interval in by_name.get(name, [])
            ]
        ),
        "candidate": interval_union_seconds(
            [
                interval
                for name in ("short_reward", "dynamic_sampling_filter")
                for interval in by_name.get(name, [])
            ]
        ),
    }
    if any(value <= 0 for value in seconds.values()):
        raise ValueError(f"calibration stage interval is unavailable: {seconds}")
    utilization = [float(value) for value in gpu_samples.get("gpu_utilization_percent", [])]
    if not utilization:
        raise ValueError("calibration requires sampled GPU utilization")
    return {
        "schema_version": "qwen3-1p7b-node-calibration-v1",
        "normal_decode_tokens_per_second": normal_tokens / seconds["normal"],
        "reward_full_response_tokens_per_second": normal_tokens / seconds["reward"],
        "actor_valid_tokens_per_second": actor_tokens / seconds["actor"],
        "candidate_batches_per_second": candidate_batches / seconds["candidate"],
        "gpu_utilization_mean": sum(utilization) / len(utilization),
        "gpu_utilization_sample_count": len(utilization),
        "workload": {
            "normal_decode_tokens": normal_tokens,
            "actor_valid_tokens": actor_tokens,
            "candidate_batches": candidate_batches,
        },
        "stage_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-jsonl", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in args.profile_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(records) != 1:
        raise SystemExit(f"calibration requires exactly one profile step, got {len(records)}")
    result = extract(records[0], json.loads(args.gpu_samples.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
