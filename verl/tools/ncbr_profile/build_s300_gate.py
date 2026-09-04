#!/usr/bin/env python3
"""Fail closed unless every prerequisite receipt authorizes formal S300."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "calibration_workload",
    "calibration_performance",
    "gate0_baseline",
    "gate0_v1",
    "selection",
    "fixed_replay_baseline",
    "fixed_replay_v1",
    "overhead",
    "acceptance_baseline_log",
    "acceptance_baseline_checkpoint",
    "acceptance_v1_log",
    "acceptance_v1_checkpoint",
    "s300_estimate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(spec: dict[str, Any], base: Path = Path(".")) -> dict[str, Any]:
    receipts = spec["receipts"]
    missing = REQUIRED - receipts.keys()
    unexpected = receipts.keys() - REQUIRED - {"mechanism_v1"}
    if missing or unexpected:
        raise ValueError(f"S300 gate receipt set mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    loaded = {}
    provenance = {}
    for name, raw_path in receipts.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError(f"S300 prerequisite is not PASS: {name}: {path}")
        loaded[name] = payload
        provenance[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    selected = str(loaded["selection"].get("selected_candidate"))
    if selected != spec["selected_candidate"]:
        raise ValueError(f"selected candidate mismatch: {selected} != {spec['selected_candidate']}")
    selection_requires_mechanism = selected in loaded["selection"].get("mechanism_required_candidates", [])
    if bool(spec.get("mechanism_required")) != selection_requires_mechanism:
        raise ValueError("mechanism_required disagrees with the selected candidate's natural coverage")
    if selection_requires_mechanism != ("mechanism_v1" in receipts):
        raise ValueError("mechanism receipt presence does not match the selected candidate's natural coverage")
    if int(loaded["overhead"].get("fixed_replay_receipt_count", 0)) != 2:
        raise ValueError("diagnostics overhead gate must aggregate both arm fixed replays")
    code_sha = str(spec["code_sha"])
    if len(code_sha) != 40:
        raise ValueError("S300 gate code SHA must be a full 40-character commit")
    for name, payload in loaded.items():
        receipt_sha = payload.get("code_sha")
        if receipt_sha is not None and receipt_sha != code_sha:
            raise ValueError(f"S300 prerequisite code SHA mismatch: {name}: {receipt_sha}")
    return {
        "schema_version": "qwen3-1p7b-ncbr-formal-s300-gate-v1",
        "status": "PASS",
        "s300_authorized": True,
        "code_sha": code_sha,
        "selected_candidate": selected,
        "receipts": provenance,
        "scope": {
            "optimizer_steps": 300,
            "seeds": [42],
            "arms": ["baseline", "v1"],
            "step_600_authorized": False,
            "second_seed_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text(encoding="utf-8"))
    result = build(spec, args.input.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
