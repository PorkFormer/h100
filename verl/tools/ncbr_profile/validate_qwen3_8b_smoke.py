#!/usr/bin/env python3
"""Validate the independent three-step Qwen3-8B NCBR mechanism smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(summary: dict) -> dict:
    requests = int(summary.get("continuation_request_count", 0))
    checks = {
        "three_steps_complete": summary.get("completed_steps") == [1, 2, 3],
        "continuation_requests_nonzero": requests > 0,
        "no_oom": summary.get("oom_count") == 0,
        "no_deadlock": summary.get("deadlock") is False,
        "no_timeout": summary.get("timeout_count") == 0,
        "long_verifier_rows_match": summary.get("long_verifier_rows") == summary.get("expected_long_verifier_rows"),
        "prefix_penalty_drift_zero": summary.get("prefix_penalty_drift_max") == 0,
        "finite_metrics": summary.get("nan_count") == 0 and summary.get("inf_count") == 0,
        "boundary_correction_active": int(summary.get("boundary_applied_count", 0)) > 0,
        "continuation_tail_excluded_from_actor": summary.get("actor_tail_token_count") == 0,
        "teardown_pass": summary.get("teardown_status") == "PASS",
    }
    if requests == 0:
        classification = "mechanism_coverage_insufficient"
    elif all(checks.values()):
        classification = "qualified"
    else:
        classification = "system_or_invariant_failure"
    return {
        "schema_version": "qwen3-8b-ncbr-smoke-gate-v1",
        "status": "PASS" if classification == "qualified" else "FAIL",
        "classification": classification,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
