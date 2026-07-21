"""Pure Immediate Answer Prefix Probe protocol and result mapping."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from verl.utils.ray_utils import auto_await

PROMPT_TRAJECTORY_SENTINEL = "__prompt__"


@dataclass(frozen=True)
class ProbeTrajectory:
    uid: str
    trajectory_id: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ProbeTarget:
    trajectory_index: int
    position_indices: tuple[int, ...]


@dataclass(frozen=True)
class ProbeRequest:
    request_id: str
    policy_version: int
    uid: str
    trajectory_id: str
    relative_position: float
    absolute_horizon: int
    input_token_ids: tuple[int, ...]
    grouped_seed: int
    branch_count: int
    targets: tuple[ProbeTarget, ...]


@dataclass(frozen=True)
class ProbeBranchResult:
    request_id: str
    branch_id: int
    success: float | bool | None
    actual_policy_version: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProbeAggregation:
    values: tuple[tuple[float, ...], ...]
    valid_mask: tuple[tuple[bool, ...], ...]


def relative_horizons(response_length: int, positions: Sequence[float]) -> tuple[int, ...]:
    """Map relative positions to absolute token horizons with floor semantics."""
    if response_length <= 0:
        raise ValueError(f"response_length must be positive, got {response_length}")
    horizons = tuple(math.floor(float(position) * response_length) for position in positions)
    if any(horizon < 0 or horizon >= response_length for horizon in horizons):
        raise ValueError("relative positions must map into [0, response_length)")
    return horizons


def first_nonempty_line(text: str) -> str:
    """Return the first generated line containing non-whitespace text."""
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def immediate_verifier_text(candidate: str) -> str:
    return f"Answer: {candidate}"


def derive_grouped_request_seed(
    global_step: int,
    uid: str,
    trajectory_id: str,
    relative_position: float,
    ordered_branch_ids: Sequence[int],
) -> int:
    """Derive one stable seed for a grouped vLLM request without touching global RNGs."""
    payload = json.dumps(
        [int(global_step), str(uid), str(trajectory_id), float(relative_position), list(ordered_branch_ids)],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _request_id(policy_version: int, uid: str, trajectory_id: str, horizon: int) -> str:
    identity = json.dumps([policy_version, uid, trajectory_id, horizon], separators=(",", ":")).encode()
    return f"probe-{policy_version}-{hashlib.sha256(identity).hexdigest()[:20]}"


def build_probe_requests(
    trajectories: Sequence[ProbeTrajectory],
    *,
    policy_version: int,
    relative_positions: Sequence[float],
    answer_prefix_token_ids: Sequence[int],
    n: int,
    max_tokens: int,
    max_model_len: int,
    probe_zero_position: bool = True,
    strict: bool = True,
) -> list[ProbeRequest]:
    """Build deduplicated grouped requests directly from retained raw token IDs."""
    branch_ids = tuple(range(n))
    prefix_ids = tuple(int(token_id) for token_id in answer_prefix_token_ids)
    prompt_by_uid: dict[str, tuple[int, ...]] = {}
    zero_targets: dict[str, list[tuple[int, int]]] = {}
    nonzero_targets: list[tuple[int, ProbeTrajectory, int, tuple[int, ...]]] = []

    for trajectory_index, trajectory in enumerate(trajectories):
        prompt_ids = tuple(int(token_id) for token_id in trajectory.prompt_token_ids)
        if trajectory.uid in prompt_by_uid and prompt_by_uid[trajectory.uid] != prompt_ids:
            raise ValueError(f"retained prompt group {trajectory.uid!r} has inconsistent prompt token IDs")
        prompt_by_uid[trajectory.uid] = prompt_ids
        horizons = relative_horizons(len(trajectory.response_token_ids), relative_positions)
        positions_by_horizon: dict[int, list[int]] = {}
        for position_index, (position, horizon) in enumerate(zip(relative_positions, horizons, strict=True)):
            if horizon == 0:
                if probe_zero_position or float(position) != 0.0:
                    zero_targets.setdefault(trajectory.uid, []).append((trajectory_index, position_index))
            else:
                positions_by_horizon.setdefault(horizon, []).append(position_index)
        for horizon, position_indices in positions_by_horizon.items():
            nonzero_targets.append((trajectory_index, trajectory, horizon, tuple(position_indices)))

    requests: list[ProbeRequest] = []
    for uid, flattened_targets in zero_targets.items():
        grouped: dict[int, list[int]] = {}
        for trajectory_index, position_index in flattened_targets:
            grouped.setdefault(trajectory_index, []).append(position_index)
        targets = tuple(ProbeTarget(index, tuple(position_indices)) for index, position_indices in grouped.items())
        relative_position = float(relative_positions[min(index for _, index in flattened_targets)])
        input_ids = (*prompt_by_uid[uid], *prefix_ids)
        requests.append(
            _make_request(
                policy_version,
                uid,
                PROMPT_TRAJECTORY_SENTINEL,
                relative_position,
                0,
                input_ids,
                branch_ids,
                targets,
                max_tokens,
                max_model_len,
                strict,
            )
        )

    for trajectory_index, trajectory, horizon, position_indices in nonzero_targets:
        relative_position = float(relative_positions[position_indices[0]])
        input_ids = (
            *trajectory.prompt_token_ids,
            *trajectory.response_token_ids[:horizon],
            *prefix_ids,
        )
        requests.append(
            _make_request(
                policy_version,
                trajectory.uid,
                trajectory.trajectory_id,
                relative_position,
                horizon,
                input_ids,
                branch_ids,
                (ProbeTarget(trajectory_index, position_indices),),
                max_tokens,
                max_model_len,
                strict,
            )
        )
    return requests


def _make_request(
    policy_version: int,
    uid: str,
    trajectory_id: str,
    relative_position: float,
    horizon: int,
    input_ids: tuple[int, ...],
    branch_ids: tuple[int, ...],
    targets: tuple[ProbeTarget, ...],
    max_tokens: int,
    max_model_len: int,
    strict: bool,
) -> ProbeRequest:
    if len(input_ids) + max_tokens > max_model_len:
        message = (
            f"Probe context overflow: input_len={len(input_ids)} + max_tokens={max_tokens} "
            f"exceeds max_model_len={max_model_len}"
        )
        if strict:
            raise ValueError(message)
        raise ValueError(message)  # non-strict omission is added only with explicit validity accounting
    return ProbeRequest(
        request_id=_request_id(policy_version, uid, trajectory_id, horizon),
        policy_version=policy_version,
        uid=uid,
        trajectory_id=trajectory_id,
        relative_position=relative_position,
        absolute_horizon=horizon,
        input_token_ids=tuple(int(token_id) for token_id in input_ids),
        grouped_seed=derive_grouped_request_seed(policy_version, uid, trajectory_id, relative_position, branch_ids),
        branch_count=len(branch_ids),
        targets=targets,
    )


def aggregate_probe_successes(values: Mapping[int, float | bool | None], n: int, strict: bool) -> float:
    """Aggregate explicit grouped-output indices into one recoverability value."""
    expected = set(range(n))
    received = set(values)
    invalid = received - expected
    missing = expected - received
    if invalid and strict:
        raise ValueError(f"invalid Probe branch IDs: {sorted(invalid)}")
    if missing and strict:
        raise ValueError(f"missing Probe branches: {sorted(missing)}")
    valid = [float(values[index]) for index in sorted(received & expected) if values[index] is not None]
    if strict and len(valid) != n:
        raise ValueError("missing or invalid Probe branch success value")
    if not valid:
        raise ValueError("no valid Probe branches")
    return sum(valid) / len(valid)


def aggregate_probe_results(
    requests: Sequence[ProbeRequest],
    results: Iterable[ProbeBranchResult],
    *,
    trajectory_count: int,
    position_count: int,
    n: int,
    strict: bool = True,
    expected_policy_version: int | None = None,
) -> ProbeAggregation:
    """Aggregate arbitrarily ordered results by explicit request and branch IDs."""
    request_by_id = {request.request_id: request for request in requests}
    by_request: dict[str, dict[int, float | bool | None]] = {}
    versions_by_request: dict[str, set[int]] = {}
    for result in results:
        if result.request_id not in request_by_id:
            if strict:
                raise ValueError(f"unknown Probe request ID: {result.request_id}")
            continue
        if result.error is not None and strict:
            raise ValueError(f"Probe request {result.request_id} failed: {result.error}")
        if result.actual_policy_version is None:
            raise ValueError(f"Probe request {result.request_id} is missing actual policy version")
        versions_by_request.setdefault(result.request_id, set()).add(int(result.actual_policy_version))
        branches = by_request.setdefault(result.request_id, {})
        if result.branch_id in branches and strict:
            raise ValueError(f"duplicate Probe branch {result.branch_id} for {result.request_id}")
        branches[result.branch_id] = result.success

    actual_versions: set[int] = set()
    for request in requests:
        request_versions = versions_by_request.get(request.request_id, set())
        if len(request_versions) > 1:
            raise ValueError(f"Probe request {request.request_id} has mixed actual policy versions")
        actual_versions.update(request_versions)
    if len(actual_versions) > 1:
        raise ValueError("Probe requests have mixed actual policy versions")
    if requests and not actual_versions:
        raise ValueError("Probe results are missing actual policy version")
    actual_policy_version = next(iter(actual_versions), None)
    retained_policy_version = (
        int(expected_policy_version)
        if expected_policy_version is not None
        else int(requests[0].policy_version) if requests else None
    )
    if actual_policy_version != retained_policy_version:
        raise ValueError(
            f"Probe actual policy version {actual_policy_version} does not match retained rollout policy version "
            f"{retained_policy_version}"
        )
    if any(int(request.policy_version) != retained_policy_version for request in requests):
        raise ValueError("Probe requests have mixed requested policy versions")

    values = [[0.0] * position_count for _ in range(trajectory_count)]
    valid_mask = [[False] * position_count for _ in range(trajectory_count)]
    for request in requests:
        value = aggregate_probe_successes(by_request.get(request.request_id, {}), n=n, strict=strict)
        for target in request.targets:
            for position_index in target.position_indices:
                values[target.trajectory_index][position_index] = value
                valid_mask[target.trajectory_index][position_index] = True
    return ProbeAggregation(tuple(tuple(row) for row in values), tuple(tuple(row) for row in valid_mask))


@auto_await
async def generate_grouped_probe_results(
    client: Any,
    requests: Sequence[ProbeRequest],
    *,
    sampling_params: Mapping[str, Any],
    score_candidate: Callable[[ProbeRequest, str], bool | float],
) -> list[ProbeBranchResult]:
    """Generate and score grouped Probe requests without mutating rollout sampling state."""

    async def generate_one(request: ProbeRequest) -> list[ProbeBranchResult]:
        params = dict(sampling_params)
        params.update({"n": request.branch_count, "seed": request.grouped_seed})
        outputs = await client.generate_grouped(
            request.request_id,
            prompt_ids=list(request.input_token_ids),
            sampling_params=params,
        )
        results: list[ProbeBranchResult] = []
        for fallback_index, output in enumerate(outputs):
            extra_fields = output.extra_fields or {}
            branch_id = int(extra_fields.get("branch_id", fallback_index))
            candidate = first_nonempty_line(str(extra_fields.get("text", "")))
            success = score_candidate(request, immediate_verifier_text(candidate))
            actual_policy_version = extra_fields.get("global_steps")
            results.append(
                ProbeBranchResult(
                    request.request_id,
                    branch_id,
                    success,
                    actual_policy_version=(
                        int(actual_policy_version) if actual_policy_version is not None else None
                    ),
                )
            )
        return results

    grouped = await asyncio.gather(*(generate_one(request) for request in requests))
    return [result for request_results in grouped for result in request_results]
