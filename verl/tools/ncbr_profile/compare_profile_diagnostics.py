#!/usr/bin/env python3
"""Build workload-normalized diagnostics off/on stage-cost pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

UNIT_NAMES = (
    "u_request",
    "u_cont_input",
    "u_tail_decode",
    "u_long_row",
    "u_long_token",
    "u_normal",
    "u_actor",
    "u_candidate",
)


def compare(pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    costs = {}
    omitted = {}
    for label, (off, on) in pairs.items():
        if off.get("unstable") or on.get("unstable"):
            raise ValueError(f"diagnostics profile pair is unstable: {label}")
        for unit in UNIT_NAMES:
            off_value = off["stable_window_unit_cost_medians"][unit]
            on_value = on["stable_window_unit_cost_medians"][unit]
            name = f"{label}/{unit}"
            if isinstance(off_value, int | float) and isinstance(on_value, int | float):
                costs[name] = {"off": float(off_value), "on": float(on_value)}
            else:
                omitted[name] = {"off": off_value, "on": on_value, "reason": "zero_or_unavailable_workload"}
    if not costs:
        raise ValueError("no comparable workload-normalized diagnostics stage costs")
    return {
        "schema_version": "ncbr-real-profile-diagnostics-comparison-v1",
        "stage_costs": costs,
        "omitted": omitted,
        "raw_step_wall_used_for_gate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, metavar="LABEL=OFF_ANALYSIS:ON_ANALYSIS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-costs-output", type=Path, required=True)
    args = parser.parse_args()
    pairs = {}
    for item in args.pair:
        label, separator, paths = item.partition("=")
        off_path, colon, on_path = paths.partition(":")
        if not separator or not colon or not label or label in pairs:
            raise SystemExit(f"invalid or duplicate profile pair: {item}")
        pairs[label] = (
            json.loads(Path(off_path).read_text(encoding="utf-8")),
            json.loads(Path(on_path).read_text(encoding="utf-8")),
        )
    result = compare(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.stage_costs_output.parent.mkdir(parents=True, exist_ok=True)
    args.stage_costs_output.write_text(
        json.dumps(result["stage_costs"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
