#!/usr/bin/env python3
"""Apply the frozen Qwen3-8B P0/P1/P-safe profiling decision tree."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REQUIRED_STEP_FIELDS = {
    "step",
    "valid_optimizer_step",
    "total_seconds",
    "rollout_seconds",
    "actor_seconds",
    "normal_tokens_per_second",
    "candidate_batches",
}


def audit(profile: dict[str, Any], candidate: str, memory_limit_gib: float) -> dict[str, Any]:
    steps = profile.get("steps", [])
    gpu_ids = set(profile.get("peak_nvml_memory_gib", {}))
    required_gpus = {str(index) for index in range(16)}
    all_peaks = [
        value
        for name in ("peak_nvml_memory_gib", "peak_allocated_gib", "peak_reserved_gib")
        for value in profile.get(name, {}).values()
    ]
    checks = {
        "candidate_identity": profile.get("candidate") == candidate,
        "five_valid_steps": len(steps) == 5
        and [row.get("step") for row in steps] == [1, 2, 3, 4, 5]
        and all(row.get("valid_optimizer_step") is True for row in steps),
        "complete_step_metrics": all(REQUIRED_STEP_FIELDS <= row.keys() for row in steps),
        "sixteen_gpu_peaks": gpu_ids == required_gpus
        and set(profile.get("peak_allocated_gib", {})) == required_gpus
        and set(profile.get("peak_reserved_gib", {})) == required_gpus,
        "one_second_gpu_sampling": profile.get("gpu_sample_interval_seconds") == 1
        and set(profile.get("gpu_utilization_distribution", {})) == required_gpus,
        "memory_limit": len(all_peaks) == 48 and max(all_peaks, default=float("inf")) <= memory_limit_gib,
        "no_oom": profile.get("oom_count") == 0,
        "no_worker_loss": profile.get("worker_loss_count") == 0,
        "no_preemption": profile.get("preemption_count") == 0,
        "no_deadlock": profile.get("deadlock") is False,
        "vllm_scheduling_present": isinstance(profile.get("vllm_scheduling"), dict)
        and profile.get("vllm_scheduling", {}).get("preemptions") == 0,
        "ray_worker_status_present": isinstance(profile.get("ray_worker_status"), dict)
        and profile.get("ray_worker_status", {}).get("lost") == 0,
    }
    measured = [float(row["total_seconds"]) for row in steps[1:]] if checks["five_valid_steps"] else []
    return {
        "candidate": candidate,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "peak_nvml_memory_gib": max(profile.get("peak_nvml_memory_gib", {}).values(), default=None),
        "step_2_5_median_total_seconds": statistics.median(measured) if measured else None,
        "metrics": profile,
    }


def select(p0: dict[str, Any], p1: dict[str, Any] | None, p_safe: dict[str, Any] | None) -> dict[str, Any]:
    p0_audit = audit(p0, "P0", 36.0)
    audits = {"P0": p0_audit}
    selected = None
    reason = None
    if p0_audit["status"] == "PASS":
        if p1 is None:
            reason = "P0 passed, so P1 profiling is required before selection"
        else:
            p1_audit = audit(p1, "P1", 38.5)
            audits["P1"] = p1_audit
            ratio_ok = (
                p1_audit["step_2_5_median_total_seconds"] is not None
                and p1_audit["step_2_5_median_total_seconds"]
                <= 1.10 * p0_audit["step_2_5_median_total_seconds"]
            )
            p1_audit["checks"]["not_over_10_percent_slower_than_p0"] = ratio_ok
            p1_audit["status"] = "PASS" if all(p1_audit["checks"].values()) else "FAIL"
            selected = "P1" if p1_audit["status"] == "PASS" else "P0"
            reason = "P1 passed every hard gate" if selected == "P1" else "P1 failed; freeze P0"
    elif p_safe is None:
        reason = "P0 failed, so P_SAFE profiling is required"
    else:
        safe_audit = audit(p_safe, "P_SAFE", 38.5)
        audits["P_SAFE"] = safe_audit
        if safe_audit["status"] == "PASS":
            selected = "P_SAFE"
            reason = "P0 failed and P_SAFE passed"
        else:
            reason = "P0 and P_SAFE both failed; training hyperparameters must not be changed"
    return {
        "schema_version": "qwen3-8b-ncbr-profile-selection-v1",
        "status": "PASS" if selected else "FAIL",
        "selected_candidate": selected,
        "reason": reason,
        "audits": audits,
        "selection_uses_reward_or_accuracy": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path)
    parser.add_argument("--p-safe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: None if path is None else json.loads(path.read_text(encoding="utf-8"))
    result = select(load(args.p0), load(args.p1), load(args.p_safe))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_candidate": result["selected_candidate"]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
