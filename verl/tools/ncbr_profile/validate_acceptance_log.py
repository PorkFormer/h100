#!/usr/bin/env python3
"""Attest Step 0/5 validation and benchmark separation in an acceptance log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    validation_steps = {
        str(step): bool(re.search(rf"test_gen_batch meta info:.*['\"]global_steps['\"]:\s*{step}\b", text))
        for step in (0, 5)
    }
    step_five_lines = [line for line in text.splitlines() if "step:5 " in line and "val-" in line]
    benchmark_metrics = {
        "AIME2024": any("aime2024" in line.lower() or "aime-2024" in line.lower() for line in step_five_lines),
        "AIME2025": any("aime2025" in line.lower() or "aime-2025" in line.lower() for line in step_five_lines),
    }
    checks = {
        "step_0_validation": validation_steps["0"],
        "step_5_validation": validation_steps["5"],
        "AIME2024_reported_separately": benchmark_metrics["AIME2024"],
        "AIME2025_reported_separately": benchmark_metrics["AIME2025"],
    }
    result = {
        "schema_version": "ncbr-five-step-acceptance-log-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(f"acceptance validation-log gate failed: {checks}")


if __name__ == "__main__":
    main()
