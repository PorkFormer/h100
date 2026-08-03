"""Simulate OBCF signal density from matched current-policy prefix scores."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq
import torch

from verl.experimental.on_policy_budgeted_capability_floor.cache import (
    CacheExpectations,
    CapabilityFloorCache,
)
from verl.experimental.on_policy_budgeted_capability_floor.math import (
    compute_capability_advantage,
    compute_capability_floor,
    summarize_floor_actionability,
)

ENGINEERING_GATE_THRESHOLD = 0.05


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _strict_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def simulate_obcf_signal(
    *,
    cache_rows: Iterable[Mapping[str, Any]],
    current_score_rows: Iterable[Mapping[str, Any]],
    reference_budget: int,
    current_rollouts_per_prompt: int,
    train_batch_size: int,
    model_id: str,
    override_reference_tolerance_count: int | None = None,
) -> dict[str, Any]:
    """Return exact prompt-level OBCF coverage and deficit diagnostics."""
    reference_budget = _strict_int(reference_budget, "reference_budget")
    current_rollouts_per_prompt = _strict_int(
        current_rollouts_per_prompt, "current_rollouts_per_prompt"
    )
    train_batch_size = _strict_int(train_batch_size, "train_batch_size")
    if reference_budget <= 0 or current_rollouts_per_prompt <= 0 or train_batch_size <= 0:
        raise ValueError("budget, rollout count, and train batch size must be positive")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    cache = [dict(row) for row in cache_rows]
    if not cache:
        raise ValueError("cache must contain protected prompts")
    if override_reference_tolerance_count is not None:
        override_reference_tolerance_count = _strict_int(
            override_reference_tolerance_count,
            "override_reference_tolerance_count",
        )
        if override_reference_tolerance_count < 0:
            raise ValueError("override_reference_tolerance_count must be nonnegative")
    inferred_tolerances: set[int] = set()
    base_rollout_counts: set[int] = set()
    for row in cache:
        base_count = _strict_int(
            row.get("base_prefix_success_count"),
            "base_prefix_success_count",
        )
        base_rollout_count = _strict_int(
            row.get("base_rollout_count"),
            "base_rollout_count",
        )
        floor_count = _strict_int(row.get("floor_count"), "floor_count")
        if not 0 <= floor_count <= base_count <= base_rollout_count:
            raise ValueError("cache floor counts are outside exact Base rollout bounds")
        inferred_tolerances.add(base_count - floor_count)
        base_rollout_counts.add(base_rollout_count)
        if override_reference_tolerance_count is not None:
            row["floor_count"] = max(
                base_count - override_reference_tolerance_count,
                0,
            )
            row["capability_floor"] = compute_capability_floor(
                base_success_count=base_count,
                base_rollout_count=base_rollout_count,
                tolerance_count=override_reference_tolerance_count,
            )
    if len(base_rollout_counts) != 1:
        raise ValueError("cache rows must use one base_rollout_count")
    if override_reference_tolerance_count is None:
        if len(inferred_tolerances) != 1:
            raise ValueError("cache rows do not imply one reference_tolerance_count")
        reference_tolerance_count = next(iter(inferred_tolerances))
        floor_source = "cache"
    else:
        reference_tolerance_count = override_reference_tolerance_count
        floor_source = "override"
    actionability = summarize_floor_actionability(
        cache_rows=cache,
        current_rollouts_per_prompt=current_rollouts_per_prompt,
    )
    protected_by_id: dict[int, dict[str, Any]] = {}
    for row in cache:
        prompt_id = _strict_int(row.get("prompt_id"), "cache prompt_id")
        if prompt_id in protected_by_id:
            raise ValueError("duplicate protected prompt identity")
        protected_by_id[prompt_id] = row

    prefix_field = f"prefix_reward_{reference_budget}"
    error_field = f"prefix_error_{reference_budget}"
    token_count_field = f"prefix_token_count_{reference_budget}"
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, int, int]] = set()
    for source in current_score_rows:
        row = dict(source)
        if row.get("model_id") != model_id:
            raise ValueError("current score model_id mismatch")
        prompt_id = _strict_int(row.get("prompt_id"), "prompt_id")
        rollout_index = _strict_int(row.get("rollout_index"), "rollout_index")
        identity = (model_id, prompt_id, rollout_index)
        if identity in identities:
            raise ValueError(f"duplicate current rollout identity {identity}")
        identities.add(identity)
        if prefix_field not in row:
            raise ValueError(f"current score is missing {prefix_field}")
        if not isinstance(row[prefix_field], bool):
            raise ValueError(f"{prefix_field} must be boolean")
        if error_field not in row or row[error_field]:
            raise ValueError(f"{error_field} must be present and empty")
        prefix_token_count = _strict_int(row.get(token_count_field), token_count_field)
        response_token_count = _strict_int(
            row.get("response_token_count"), "response_token_count"
        )
        if not 0 <= prefix_token_count <= min(reference_budget, response_token_count):
            raise ValueError(f"{token_count_field} is outside the exact response bounds")
        grouped[prompt_id].append(row)
    if not grouped:
        raise ValueError("current score rows must be nonempty")

    for prompt_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["rollout_index"]))
        if [int(row["rollout_index"]) for row in rows] != list(
            range(current_rollouts_per_prompt)
        ):
            raise ValueError(f"prompt {prompt_id} rollout indices/count are not exact")
        prompt_hashes = {str(row.get("prompt_hash", "")) for row in rows}
        if len(prompt_hashes) != 1 or not next(iter(prompt_hashes)):
            raise ValueError(f"prompt {prompt_id} has inconsistent prompt_hash values")
        protected = protected_by_id.get(prompt_id)
        if protected is not None and any(
            str(row.get("prompt_hash", "")) != str(protected["prompt_hash"])
            for row in rows
        ):
            raise ValueError("current prompt_hash does not match protected cache")
    missing = sorted(set(protected_by_id) - grouped.keys())
    if missing:
        raise ValueError(f"current scores are missing protected prompt groups {missing}")

    protected_ids = list(protected_by_id)
    rewards: list[float] = []
    token_presence: list[bool] = []
    group_ids: list[int] = []
    floors: list[float] = []
    for group_id, prompt_id in enumerate(protected_ids):
        rows = grouped[prompt_id]
        rewards.extend(float(row[prefix_field]) for row in rows)
        token_presence.extend(bool(row[token_count_field]) for row in rows)
        group_ids.extend([group_id] * current_rollouts_per_prompt)
        floors.append(float(protected_by_id[prompt_id]["capability_floor"]))
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    result = compute_capability_advantage(
        prefix_rewards=reward_tensor,
        group_ids=torch.tensor(group_ids, dtype=torch.long),
        capability_floors=torch.tensor(floors, dtype=torch.float32),
        response_mask=torch.tensor(token_presence, dtype=torch.bool).unsqueeze(1),
        reference_budget=1,
    )
    rollout_nonzero = result.token_advantage.ne(0).any(dim=1).to(torch.long)
    group_nonzero = torch.zeros(len(protected_ids), dtype=torch.long)
    group_nonzero.scatter_add_(0, torch.tensor(group_ids), rollout_nonzero)
    active_without_gradient = result.active_group & (group_nonzero == 0)

    strata: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    mixed = (result.q_current > 0) & (result.q_current < 1)
    all_zero = result.q_current == 0
    nonzero_gradient = group_nonzero > 0
    for group_id, prompt_id in enumerate(protected_ids):
        base_count = int(protected_by_id[prompt_id]["base_prefix_success_count"])
        values = strata[base_count]
        values["capability_floor"].append(floors[group_id])
        values["q_current"].append(float(result.q_current[group_id].item()))
        values["deficit"].append(float(result.deficit[group_id].item()))
        values["active"].append(float(result.active_group[group_id].item()))
        values["mixed"].append(float(mixed[group_id].item()))
        values["all_zero"].append(float(all_zero[group_id].item()))
        values["nonzero_gradient"].append(float(nonzero_gradient[group_id].item()))
        values["active_without_gradient"].append(
            float(active_without_gradient[group_id].item())
        )
        row = protected_by_id[prompt_id]
        structurally_inert = not (
            int(row["floor_count"]) * current_rollouts_per_prompt
            > int(row["base_rollout_count"])
        )
        values["structurally_inert"].append(float(structurally_inert))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    stratum_report = {
        str(count): {
            "prompt_count": len(values["q_current"]),
            "capability_floor": mean(values["capability_floor"]),
            "q_current_mean": mean(values["q_current"]),
            "deficit_mean": mean(values["deficit"]),
            "active_fraction": mean(values["active"]),
            "mixed_fraction": mean(values["mixed"]),
            "all_zero_fraction": mean(values["all_zero"]),
            "nonzero_gradient_fraction": mean(values["nonzero_gradient"]),
            "active_without_gradient_fraction": mean(values["active_without_gradient"]),
            "structurally_inert_fraction": mean(values["structurally_inert"]),
        }
        for count, values in sorted(strata.items())
    }
    nonzero_fraction = float(result.nonzero_gradient_group_fraction.item())
    actionability_passed = actionability.inert_prompt_count == 0
    signal_density_passed = nonzero_fraction >= ENGINEERING_GATE_THRESHOLD
    report: dict[str, Any] = {
        "floor_configuration": {
            "source": floor_source,
            "reference_tolerance_count": reference_tolerance_count,
            "current_rollouts_per_prompt": current_rollouts_per_prompt,
            "minimum_positive_empirical_rate": (
                actionability.minimum_positive_empirical_rate
            ),
        },
        "actionability": {
            "protected_prompt_count": actionability.protected_prompt_count,
            "actionable_prompt_count": actionability.actionable_prompt_count,
            "inert_prompt_count": actionability.inert_prompt_count,
            "inert_prompt_fraction": actionability.inert_prompt_fraction,
            "passed": actionability_passed,
        },
        "by_base_success_count": stratum_report,
        "protected_prompt_count": len(protected_ids),
        "current_prompt_count": len(grouped),
        "q_current_mean": float(result.q_current.mean().item()),
        "floor_mean": float(torch.tensor(floors).mean().item()),
        "deficit_mean": float(result.observed_constraint.item()),
        "active_group_fraction": float(result.active_group.float().mean().item()),
        "mixed_group_fraction": float(result.mixed_group_fraction.item()),
        "all_zero_group_fraction": float(result.all_zero_group_fraction.item()),
        "all_one_group_fraction": float(result.all_one_group_fraction.item()),
        "nonzero_gradient_fraction": nonzero_fraction,
        "active_without_gradient_fraction": float(active_without_gradient.float().mean().item()),
        "expected_prefix_verifier_calls_per_step": float(
            train_batch_size
            * (len(protected_ids) / len(grouped))
            * current_rollouts_per_prompt
        ),
        "engineering_gates": {
            "actionability": {
                "passed": actionability_passed,
                "criterion": "inert_prompt_count == 0",
            },
            "signal_density": {
                "passed": signal_density_passed,
                "threshold": ENGINEERING_GATE_THRESHOLD,
                "metric": "nonzero_gradient_fraction among protected prompts",
            },
            "passed": actionability_passed and signal_density_passed,
        },
    }

    protected_current_rows = [row for prompt_id in protected_ids for row in grouped[prompt_id]]
    full_present = ["full_reward" in row for row in protected_current_rows]
    if any(full_present) and not all(full_present):
        raise ValueError("full_reward must be present for every row when breakdown is requested")
    if full_present and all(full_present):
        categories = {"retained": 0, "delayed": 0, "lost": 0}
        for prompt_id in protected_ids:
            rows = grouped[prompt_id]
            if any(not isinstance(row["full_reward"], bool) for row in rows):
                raise ValueError("full_reward must be boolean for optional breakdown")
            if any("full_error" not in row or row["full_error"] for row in rows):
                raise ValueError("full_error must be present and empty for optional breakdown")
            prefix_any = any(bool(row[prefix_field]) for row in rows)
            full_any = any(bool(row["full_reward"]) for row in rows)
            if prefix_any:
                categories["retained"] += 1
            elif full_any:
                categories["delayed"] += 1
            else:
                categories["lost"] += 1
        if sum(categories.values()) != len(protected_ids):
            raise AssertionError("retained/delayed/lost must partition protected prompts")
        report["retained_delayed_lost"] = {
            name: {"count": count, "fraction": count / len(protected_ids)}
            for name, count in categories.items()
        }
    return report


def engineering_exit_code(report: Mapping[str, Any]) -> int:
    """Map structural and density gates to stable CLI exit codes."""
    try:
        gates = report["engineering_gates"]
        actionability_passed = gates["actionability"]["passed"] is True
        density_passed = gates["signal_density"]["passed"] is True
    except (KeyError, TypeError) as error:
        raise ValueError("report is missing engineering gate results") from error
    if not actionability_passed:
        return 3
    if not density_passed:
        return 2
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        parts = sorted(path.glob("part-*.parquet"))
        if not parts:
            parts = sorted(path.glob("*.parquet"))
        if not parts:
            raise ValueError(f"score directory {path} contains no parquet parts")
        rows: list[dict[str, Any]] = []
        expected_schema = None
        for part in parts:
            table = pq.read_table(part)
            if expected_schema is None:
                expected_schema = table.schema
            elif table.schema != expected_schema:
                raise ValueError("partitioned score artifacts have incompatible schemas")
            rows.extend(table.to_pylist())
        return rows
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("score artifact must contain a row list")
    return payload


def main() -> None:
    parser = _ArgumentParser(
        description=(
            "Simulate OBCF signal and apply a 5% nonzero-gradient engineering gate "
            "(not a theoretical threshold)."
        )
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--current-scores", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--current-rollouts-per-prompt", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--override-reference-tolerance-count", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.cache_path / "manifest.json").read_text())
    cache = CapabilityFloorCache.load(
        args.cache_path,
        CacheExpectations(
            reference_budget=manifest["reference_budget"],
            base_rollouts_per_prompt=manifest["base_rollouts_per_prompt"],
            support_threshold=manifest["support_threshold"],
            reference_tolerance_count=manifest["reference_tolerance_count"],
            tokenizer_fingerprint=manifest["tokenizer_fingerprint"],
            chat_template_fingerprint=manifest["chat_template_fingerprint"],
            verifier_fingerprint=manifest["verifier_fingerprint"],
        ),
    )
    report = simulate_obcf_signal(
        cache_rows=cache.prompts,
        current_score_rows=_load_rows(args.current_scores),
        reference_budget=manifest["reference_budget"],
        current_rollouts_per_prompt=args.current_rollouts_per_prompt,
        train_batch_size=args.train_batch_size,
        model_id=args.model_id,
        override_reference_tolerance_count=args.override_reference_tolerance_count,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    exit_code = engineering_exit_code(report)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
