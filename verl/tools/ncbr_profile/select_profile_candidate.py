#!/usr/bin/env python3
"""Select P0/P1/P2 from component-normalized Baseline/NCBR profiles."""

from __future__ import annotations

import argparse
import json
import math
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
CANDIDATES = ("P0", "P1", "P2")
ARMS = ("baseline", "v1")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("scenario workload sample is empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stable_records(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    records = analysis["records"]
    expected = records[1:6] if len(records) >= 6 else records[1:4]
    if len(expected) < 3:
        raise ValueError("profile requires warmup Step 1 plus at least Steps 2-4")
    return expected


def _load_matrix(spec: dict[str, Any], base: Path) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        loaded[candidate] = {}
        for arm in ARMS:
            entry = dict(spec["profiles"][candidate][arm])
            analysis_path = entry.pop("analysis", None)
            if analysis_path is not None:
                path = Path(analysis_path)
                if not path.is_absolute():
                    path = base / path
                entry["analysis"] = json.loads(path.read_text(encoding="utf-8"))
            else:
                entry["analysis"] = None
            mechanism_path = entry.get("mechanism_analysis")
            if mechanism_path is not None:
                mechanism_path = Path(mechanism_path)
                if not mechanism_path.is_absolute():
                    mechanism_path = base / mechanism_path
                entry["mechanism_analysis"] = json.loads(mechanism_path.read_text(encoding="utf-8"))
            loaded[candidate][arm] = entry
    return loaded


def _scenario_workloads(
    matrix: dict[str, dict[str, dict[str, Any]]], candidates: list[str], q: float, cap_rate: float
) -> dict[str, dict[str, float]]:
    samples: dict[str, dict[str, list[float]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for candidate in candidates:
            for record in _stable_records(matrix[candidate][arm]["analysis"]):
                workload = record["workloads"]
                for name in ("normal_decode_tokens", "normal_trajectories", "actor_valid_tokens", "candidate_batches"):
                    samples[arm].setdefault(name, []).append(float(workload[name]))
                requests = float(workload["request_count"])
                if arm == "v1" and requests > 0:
                    for total, per_request in (
                        ("continuation_input_tokens", "continuation_input_per_request"),
                        ("tail_decode_tokens", "tail_decode_per_request"),
                        ("long_reward_full_response_tokens", "long_tokens_per_request"),
                    ):
                        samples[arm].setdefault(per_request, []).append(float(workload[total]) / requests)
            mechanism = matrix[candidate][arm].get("mechanism_analysis")
            if arm == "v1" and mechanism is not None:
                workload = mechanism["workloads"]
                requests = float(workload["request_count"])
                if requests <= 0:
                    raise ValueError("mechanism panel request count must be positive")
                for total, per_request in (
                    ("continuation_input_tokens", "continuation_input_per_request"),
                    ("tail_decode_tokens", "tail_decode_per_request"),
                    ("long_reward_full_response_tokens", "long_tokens_per_request"),
                ):
                    samples[arm].setdefault(per_request, []).append(float(workload[total]) / requests)
    result: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        normal_trajectories = _quantile(samples[arm]["normal_trajectories"], q)
        requests = 0.0 if arm == "baseline" else cap_rate * normal_trajectories
        values = {
            "request_count": requests,
            "continuation_input_tokens": 0.0,
            "tail_decode_tokens": 0.0,
            "long_reward_rows": requests,
            "long_reward_full_response_tokens": 0.0,
            "normal_decode_tokens": _quantile(samples[arm]["normal_decode_tokens"], q),
            "actor_valid_tokens": _quantile(samples[arm]["actor_valid_tokens"], q),
            "candidate_batches": min(_quantile(samples[arm]["candidate_batches"], q), 10.0),
            "normal_trajectories": normal_trajectories,
        }
        if arm == "v1":
            required = ("continuation_input_per_request", "tail_decode_per_request", "long_tokens_per_request")
            missing = [name for name in required if not samples[arm].get(name)]
            if missing:
                raise ValueError(f"V1 profile has no natural request workload samples: {missing}")
            values["continuation_input_tokens"] = requests * _quantile(samples[arm][required[0]], q)
            values["tail_decode_tokens"] = requests * _quantile(samples[arm][required[1]], q)
            values["long_reward_full_response_tokens"] = requests * _quantile(samples[arm][required[2]], q)
        result[arm] = values
    return result


def _predict(
    analysis: dict[str, Any],
    workloads: dict[str, float],
    factors: dict[str, float],
    mechanism_analysis: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    units = dict(analysis["stable_window_unit_cost_medians"])
    if mechanism_analysis is not None:
        for name, value in mechanism_analysis["unit_costs"].items():
            if not isinstance(units.get(name), int | float):
                units[name] = value
    denominators = {
        "u_request": "request_count",
        "u_cont_input": "continuation_input_tokens",
        "u_tail_decode": "tail_decode_tokens",
        "u_long_row": "long_reward_rows",
        "u_long_token": "long_reward_full_response_tokens",
        "u_normal": "normal_decode_tokens",
        "u_actor": "actor_valid_tokens",
        "u_candidate": "candidate_batches",
    }
    raw_total = 0.0
    normalized_total = 0.0
    components = {}
    for name in UNIT_NAMES:
        workload = workloads[denominators[name]]
        unit = units[name]
        if workload == 0:
            raw = normalized = 0.0
        else:
            if not isinstance(unit, int | float):
                raise ValueError(f"required unit cost is unavailable: {name}")
            factor = float(factors[name])
            if factor <= 0:
                raise ValueError(f"node normalization factor must be positive: {name}")
            raw = float(unit) * workload
            normalized = raw * factor
        components[name] = {
            "workload": workload,
            "unit_cost_raw": unit,
            "raw_seconds": raw,
            "normalized_seconds": normalized,
        }
        raw_total += raw
        normalized_total += normalized
    return raw_total, normalized_total, components


def select(spec: dict[str, Any], base: Path = Path(".")) -> dict[str, Any]:
    matrix = _load_matrix(spec, base)
    excluded: dict[str, list[str]] = {}
    profiled_candidates = []
    for candidate in CANDIDATES:
        reasons = []
        for arm in ARMS:
            entry = matrix[candidate][arm]
            analysis = entry.get("analysis")
            if entry.get("safety_status") != "PASS":
                reasons.append(f"{arm}:safety")
            if not isinstance(analysis, dict):
                reasons.append(f"{arm}:analysis_unavailable")
                continue
            record_count = len(analysis.get("records", []))
            if record_count not in {4, 6}:
                reasons.append(f"{arm}:incomplete_profile")
            if analysis.get("extension_required"):
                reasons.append(f"{arm}:required_extension_missing")
            if analysis.get("unstable"):
                reasons.append(f"{arm}:unstable")
            if record_count in {4, 6} and any(
                record.get("valid") is False for record in _stable_records(analysis)
            ):
                reasons.append(f"{arm}:invalid_timer_dag")
            if arm == "v1" and analysis.get("mechanism_coverage_insufficient"):
                if entry.get("mechanism_status") != "PASS":
                    reasons.append(f"{arm}:mechanism_coverage")
        if reasons:
            excluded[candidate] = sorted(set(reasons))
        else:
            profiled_candidates.append(candidate)
    if not profiled_candidates:
        raise ValueError("all candidates failed safety, stability, timer, or mechanism gates")
    scenarios = {
        "moderate": _scenario_workloads(matrix, profiled_candidates, 0.50, 0.10),
        "stress": _scenario_workloads(matrix, profiled_candidates, 0.90, 0.30),
    }
    predictions: dict[str, Any] = {}
    for candidate in profiled_candidates:
        predictions[candidate] = {}
        reasons: list[str] = []
        for arm in ARMS:
            entry = matrix[candidate][arm]
            analysis = entry["analysis"]
            factors = spec["node_unit_cost_factors"][entry["node"]]
            predictions[candidate][arm] = {}
            for scenario_name, scenario in scenarios.items():
                try:
                    raw, normalized, components = _predict(
                        analysis,
                        scenario[arm],
                        factors,
                        entry.get("mechanism_analysis"),
                    )
                except ValueError as error:
                    reasons.append(f"{arm}:{scenario_name}:{error}")
                    continue
                predictions[candidate][arm][scenario_name] = {
                    "raw_seconds": raw,
                    "normalized_seconds": normalized,
                    "components": components,
                }
        if reasons:
            excluded[candidate] = sorted(set(reasons))

    initially_eligible = [candidate for candidate in profiled_candidates if candidate not in excluded]
    if not initially_eligible:
        raise ValueError("all candidates failed safety, stability, or unit-cost availability")
    fastest_baseline = min(
        predictions[candidate]["baseline"]["moderate"]["normalized_seconds"] for candidate in initially_eligible
    )
    eligible = []
    for candidate in initially_eligible:
        baseline = predictions[candidate]["baseline"]["moderate"]["normalized_seconds"]
        if baseline > fastest_baseline * 1.10:
            excluded.setdefault(candidate, []).append("baseline_more_than_10_percent_slower")
        else:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("no candidate remains after the Baseline 10% gate")

    fastest_v1 = {
        scenario: min(predictions[candidate]["v1"][scenario]["normalized_seconds"] for candidate in eligible)
        for scenario in scenarios
    }
    scores = {}
    for candidate in eligible:
        regrets = {
            scenario: predictions[candidate]["v1"][scenario]["normalized_seconds"] / fastest_v1[scenario] - 1.0
            for scenario in scenarios
        }
        scores[candidate] = {"regret": regrets, "score": max(regrets.values())}
    best_score = min(record["score"] for record in scores.values())
    tied = [candidate for candidate in eligible if scores[candidate]["score"] <= best_score + 0.05]

    def tie_key(candidate: str) -> tuple[Any, ...]:
        system = spec["systems"][candidate]
        return (
            float(system["fixed_workload_peak_memory_bytes"]),
            int(system.get("retry_warning_count", 0)),
            int(system["tensor_model_parallel_size"]),
            not bool(system["optimizer_offload"]),
            not bool(system["ref_param_offload"]),
            int(system["max_num_seqs"]),
            float(system["gpu_memory_utilization"]),
        )

    selected = min(tied, key=tie_key)
    mechanism_required_candidates = [
        candidate
        for candidate in profiled_candidates
        if bool(matrix[candidate]["v1"]["analysis"].get("mechanism_coverage_insufficient"))
    ]
    return {
        "schema_version": "qwen3-1p7b-ncbr-profile-selection-v1",
        "status": "PASS",
        "scenario_workloads": scenarios,
        "predictions": predictions,
        "excluded": excluded,
        "fastest_baseline_normalized_seconds": fastest_baseline,
        "fastest_v1_normalized_seconds": fastest_v1,
        "scores": scores,
        "score_tie_candidates_within_5_percentage_points": tied,
        "selected_candidate": selected,
        "mechanism_required_candidates": mechanism_required_candidates,
        "selection_order": [
            "safety_and_stability",
            "baseline_within_10_percent",
            "worst_v1_moderate_stress_regret",
            "memory_warnings_tp_offload_concurrency_tiebreak",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text(encoding="utf-8"))
    result = select(spec, args.input.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
