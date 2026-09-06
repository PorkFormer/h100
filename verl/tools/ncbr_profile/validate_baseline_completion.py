#!/usr/bin/env python3
"""Validate the immutable Baseline-S300 completion receipt before NCBR starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(receipt: dict) -> None:
    required = {
        "status": "PASS",
        "arm": "baseline",
        "total_training_steps": 300,
        "completed_step": 300,
        "checkpoint_status": "PASS",
        "teardown_status": "PASS",
    }
    mismatches = {key: (value, receipt.get(key)) for key, value in required.items() if receipt.get(key) != value}
    if mismatches:
        raise SystemExit(f"Baseline completion receipt is not sufficient: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validate(json.loads(args.receipt.read_text(encoding="utf-8")))
    print(json.dumps({"status": "PASS", "receipt": str(args.receipt.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
