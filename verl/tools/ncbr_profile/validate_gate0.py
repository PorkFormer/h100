#!/usr/bin/env python3
"""Validate all three Dynamic Sampling Gate 0 cycles and stage ordering."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _validate_v1_event_order(log_text: str, expected_batches: int) -> bool:
    markers = {
        "short": "boundary_return event=short_reward_complete",
        "continuation": "boundary_return event=continuation_complete",
        "long_reward": "boundary_return event=long_reward_complete",
        "filter": "boundary_return event=filter metric=boundary_acc",
    }
    pattern = re.compile("|".join(f"(?P<{name}>{re.escape(marker)})" for name, marker in markers.items()))
    events = [match.lastgroup for match in pattern.finditer(log_text)]
    batches = 0
    cursor = 0
    while cursor < len(events):
        if events[cursor] != "short":
            return False
        cursor += 1
        if cursor < len(events) and events[cursor] == "continuation":
            cursor += 1
            if cursor >= len(events) or events[cursor] != "long_reward":
                return False
            cursor += 1
        if cursor >= len(events) or events[cursor] != "filter":
            return False
        cursor += 1
        batches += 1
    return batches == expected_batches


def validate(records: list[dict], arm: str, log_text: str) -> dict:
    checks = {
        "exactly_three_cycles": len(records) == 3,
        "cycle_ids": [record.get("cycle") for record in records] == [1, 2, 3],
        "target_cycles": all(record.get("target_cycles") == 3 for record in records),
        "candidate_batches_at_most_10": all(1 <= int(record.get("candidate_batches", 0)) <= 10 for record in records),
        "exactly_256_uid_groups": all(record.get("retained_uid_groups") == 256 for record in records),
        "exactly_2048_trajectories": all(record.get("retained_trajectories") == 2048 for record in records),
        "stopped_before_old_ref_adv_actor": all(
            record.get("stopped_before") == ["old_log_prob", "ref_log_prob", "advantage", "actor_update"]
            for record in records
        ),
        "all_cycle_status_pass": all(record.get("status") == "PASS" for record in records),
        "arm_identity": all(record.get("arm") == arm for record in records),
    }
    if arm == "baseline":
        checks["baseline_zero_continuations"] = all(
            float(record.get("boundary_return/continuation_request_count", 0.0)) == 0.0 for record in records
        )
    else:
        expected_batches = sum(int(record.get("candidate_batches", 0)) for record in records)
        checks["v1_continuation_reward_filter_order"] = _validate_v1_event_order(log_text, expected_batches)
    return {
        "schema_version": "qwen3-1p7b-dynamic-sampling-gate0-v1",
        "arm": arm,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "v1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = validate(records, args.arm, args.log.read_text(encoding="utf-8", errors="replace"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(f"Dynamic Sampling Gate 0 failed: {result['checks']}")


if __name__ == "__main__":
    main()
