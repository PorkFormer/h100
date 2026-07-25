"""Pure Immediate Answer Prefix Probe protocol and result mapping."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from verl.utils.ray_utils import auto_await

PROMPT_TRAJECTORY_SENTINEL = "__prompt__"
ProbePositionKind = Literal["relative", "absolute"]


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
    routing_key: str
    policy_version: int
    uid: str
    trajectory_id: str
    relative_position: float | None
    absolute_horizon: int
    input_token_ids: tuple[int, ...]
    grouped_seed: int
    branch_count: int
    targets: tuple[ProbeTarget, ...]
    position_kind: ProbePositionKind = "relative"


@dataclass(frozen=True)
class ProbeBranchResult:
    request_id: str
    branch_id: int
    success: float | bool | None
    actual_policy_version: int | None = None
    output_token_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ProbeAggregation:
    values: tuple[tuple[float, ...], ...]
    valid_mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True)
class AbsoluteProbePlan:
    requests: tuple[ProbeRequest, ...]
    valid_mask: tuple[tuple[bool, ...], ...]
    absolute_horizons: tuple[int, ...]


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


def derive_absolute_grouped_request_seed(
    policy_version: int,
    uid: str,
    trajectory_id: str,
    absolute_horizon: int,
    ordered_branch_ids: Sequence[int],
) -> int:
    """Derive a stable absolute-horizon seed in a separate semantic namespace."""
    payload = json.dumps(
        [
            "absolute",
            int(policy_version),
            str(uid),
            str(trajectory_id),
            int(absolute_horizon),
            list(ordered_branch_ids),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _request_id(policy_version: int, uid: str, trajectory_id: str, horizon: int) -> str:
    identity = json.dumps([policy_version, uid, trajectory_id, horizon], separators=(",", ":")).encode()
    return f"probe-{policy_version}-{hashlib.sha256(identity).hexdigest()[:20]}"


def _absolute_request_id(policy_version: int, uid: str, trajectory_id: str, horizon: int) -> str:
    identity = json.dumps(
        ["absolute", policy_version, uid, trajectory_id, horizon], separators=(",", ":")
    ).encode()
    return f"probe-absolute-{policy_version}-{hashlib.sha256(identity).hexdigest()[:20]}"


def _routing_key(policy_version: int, uid: str) -> str:
    identity = json.dumps([policy_version, uid], separators=(",", ":")).encode()
    return f"probe-route-{policy_version}-{hashlib.sha256(identity).hexdigest()[:20]}"


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


def _validate_absolute_horizons(absolute_horizons: Sequence[int]) -> tuple[int, ...]:
    if not absolute_horizons:
        raise ValueError("absolute_horizons must be nonempty")
    horizons: list[int] = []
    for horizon in absolute_horizons:
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise ValueError("absolute_horizons must contain integers")
        if horizon <= 0:
            raise ValueError("absolute_horizons must contain positive integers")
        horizons.append(horizon)
    if any(right <= left for left, right in zip(horizons, horizons[1:], strict=False)):
        raise ValueError("absolute_horizons must be strictly increasing")
    return tuple(horizons)


def build_absolute_probe_requests(
    trajectories: Sequence[ProbeTrajectory],
    *,
    trajectory_mask: Sequence[bool],
    policy_version: int,
    absolute_horizons: Sequence[int],
    answer_prefix_token_ids: Sequence[int],
    n: int,
    max_tokens: int,
    max_model_len: int,
    strict: bool = True,
) -> AbsoluteProbePlan:
    """Plan active-prefix requests at fixed absolute token horizons."""
    horizons = _validate_absolute_horizons(absolute_horizons)
    if len(trajectory_mask) != len(trajectories):
        raise ValueError("trajectory_mask length must match trajectories")
    if any(not isinstance(active, bool) for active in trajectory_mask):
        raise ValueError("trajectory_mask must contain boolean values")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    prefix_ids = tuple(int(token_id) for token_id in answer_prefix_token_ids)
    branch_ids = tuple(range(n))
    prompt_by_uid: dict[str, tuple[int, ...]] = {}
    valid_mask = [[False] * len(horizons) for _ in trajectories]
    requests: list[ProbeRequest] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        prompt_ids = tuple(int(token_id) for token_id in trajectory.prompt_token_ids)
        if trajectory.uid in prompt_by_uid and prompt_by_uid[trajectory.uid] != prompt_ids:
            raise ValueError(f"retained prompt group {trajectory.uid!r} has inconsistent prompt token IDs")
        prompt_by_uid[trajectory.uid] = prompt_ids
        if not trajectory_mask[trajectory_index]:
            continue
        response_ids = tuple(int(token_id) for token_id in trajectory.response_token_ids)
        for position_index, horizon in enumerate(horizons):
            if horizon >= len(response_ids):
                continue
            valid_mask[trajectory_index][position_index] = True
            input_ids = (*prompt_ids, *response_ids[:horizon], *prefix_ids)
            requests.append(
                _make_absolute_request(
                    policy_version=policy_version,
                    uid=trajectory.uid,
                    trajectory_id=trajectory.trajectory_id,
                    horizon=horizon,
                    input_ids=input_ids,
                    branch_ids=branch_ids,
                    target=ProbeTarget(trajectory_index, (position_index,)),
                    max_tokens=max_tokens,
                    max_model_len=max_model_len,
                    strict=strict,
                )
            )
    return AbsoluteProbePlan(
        requests=tuple(requests),
        valid_mask=tuple(tuple(row) for row in valid_mask),
        absolute_horizons=horizons,
    )


def _make_absolute_request(
    *,
    policy_version: int,
    uid: str,
    trajectory_id: str,
    horizon: int,
    input_ids: tuple[int, ...],
    branch_ids: tuple[int, ...],
    target: ProbeTarget,
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
        raise ValueError(message)
    return ProbeRequest(
        request_id=_absolute_request_id(policy_version, uid, trajectory_id, horizon),
        routing_key=_routing_key(policy_version, uid),
        policy_version=policy_version,
        uid=uid,
        trajectory_id=trajectory_id,
        relative_position=None,
        absolute_horizon=horizon,
        input_token_ids=tuple(int(token_id) for token_id in input_ids),
        grouped_seed=derive_absolute_grouped_request_seed(
            policy_version, uid, trajectory_id, horizon, branch_ids
        ),
        branch_count=len(branch_ids),
        targets=(target,),
        position_kind="absolute",
    )


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
        routing_key=_routing_key(policy_version, uid),
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
    max_concurrent_requests: int = 128,
    request_batch_size: int = 512,
) -> list[ProbeBranchResult]:
    """Generate and score grouped Probe requests without mutating rollout sampling state."""
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be positive")
    if request_batch_size <= 0:
        raise ValueError("request_batch_size must be positive")
    if request_batch_size < max_concurrent_requests:
        raise ValueError("request_batch_size must be at least max_concurrent_requests")
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def generate_one(request: ProbeRequest) -> list[ProbeBranchResult]:
        async with semaphore:
            params = dict(sampling_params)
            params.update({"n": request.branch_count, "seed": request.grouped_seed})
            outputs = await client.generate_grouped(
                request.request_id,
                prompt_ids=list(request.input_token_ids),
                sampling_params=params,
                routing_key=request.routing_key,
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
                    output_token_count=len(output.token_ids),
                )
            )
        return results

    results: list[ProbeBranchResult] = []
    for start in range(0, len(requests), request_batch_size):
        chunk = requests[start : start + request_batch_size]
        tasks = [asyncio.create_task(generate_one(request)) for request in chunk]
        try:
            grouped = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        results.extend(result for request_results in grouped for result in request_results)
    return results
