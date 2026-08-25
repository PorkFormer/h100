"""Strict natural-continuation request construction and result mapping."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from verl import DataProto
from verl.trainer.ppo.forced_answer_probe import detect_hit_response_cap
from verl.utils.ray_utils import auto_await


@dataclass(frozen=True)
class BoundaryContinuationRequest:
    parent_index: int
    request_id: str
    routing_key: str
    policy_version: int
    uid: str
    trajectory_id: str
    prompt_token_ids: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    input_token_ids: tuple[int, ...]
    max_tokens: int
    seed: int
    branch_count: int = 1


@dataclass(frozen=True)
class BoundaryContinuationBranchResult:
    request_id: str
    branch_id: int
    tail_token_ids: tuple[int, ...]
    actual_policy_version: int | None


@dataclass(frozen=True)
class BoundaryContinuationGeneration:
    parent_index: int
    request_id: str
    branch_id: int
    uid: str
    trajectory_id: str
    prompt_token_ids: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    tail_token_ids: tuple[int, ...]
    actual_policy_version: int


@dataclass(frozen=True)
class BoundaryContinuationCapture:
    hit_response_cap: np.ndarray
    requests: tuple[BoundaryContinuationRequest, ...]
    generations: tuple[BoundaryContinuationGeneration, ...]
    normal_response_tokens: int


def _stable_digest(parts: Sequence[Any]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def derive_continuation_seed(
    base_seed: int,
    policy_version: int,
    uid: str,
    trajectory_id: str,
) -> int:
    """Use a dedicated stable SHA namespace without drawing global RNG state."""
    digest = _stable_digest(
        ["natural-continuation-boundary-return", int(base_seed), int(policy_version), str(uid), str(trajectory_id)]
    )
    return int.from_bytes(bytes.fromhex(digest[:16]), "big") % (2**31)


def _request_id(policy_version: int, uid: str, trajectory_id: str) -> str:
    digest = _stable_digest(["boundary-return-request", int(policy_version), str(uid), str(trajectory_id)])
    return f"boundary-return-{int(policy_version)}-{digest[:20]}"


def _routing_key(policy_version: int, uid: str) -> str:
    digest = _stable_digest(["boundary-return-route", int(policy_version), str(uid)])
    return f"boundary-return-route-{int(policy_version)}-{digest[:20]}"


def _validated_policy_versions(batch: DataProto, expected: int) -> None:
    versions = batch.non_tensor_batch.get("rollout_policy_version")
    if versions is None or len(versions) != len(batch):
        raise ValueError("normal rollout is missing actual policy version")
    normalized: list[int] = []
    for value in np.asarray(versions, dtype=object).tolist():
        if value is None:
            raise ValueError("normal rollout is missing actual policy version")
        try:
            normalized.append(int(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"normal rollout has invalid actual policy version {value!r}") from error
    if len(set(normalized)) != 1:
        raise ValueError("normal rollout has mixed actual policy versions")
    if normalized and normalized[0] != int(expected):
        raise ValueError(
            f"normal rollout actual policy version {normalized[0]} does not match published version {expected}"
        )


def build_continuation_requests(
    rollout_batch: DataProto,
    *,
    policy_version: int,
    short_response_length: int,
    long_response_length: int,
    max_model_len: int,
    eos_token_id: int | None,
    base_seed: int,
) -> tuple[list[BoundaryContinuationRequest], np.ndarray]:
    """Build K=1 requests from the exact valid prompt and complete short prefix."""
    if long_response_length <= short_response_length:
        raise ValueError("boundary_return requires long_response_length > short response length")
    _validated_policy_versions(rollout_batch, int(policy_version))
    prompt_width = rollout_batch.batch["prompts"].shape[-1]
    prompt_mask = rollout_batch.batch["attention_mask"][:, :prompt_width].bool()
    response_mask = rollout_batch.batch["attention_mask"][:, prompt_width:].bool()
    response_lengths = response_mask.sum(-1).cpu().tolist()
    response_token_ids = [
        rollout_batch.batch["responses"][row][response_mask[row]].detach().cpu().tolist()
        for row in range(len(rollout_batch))
    ]
    finish_reasons = rollout_batch.non_tensor_batch.get("finish_reason")
    hit_cap = detect_hit_response_cap(
        finish_reasons=finish_reasons,
        response_lengths=response_lengths,
        max_response_length=short_response_length,
        response_token_ids=response_token_ids,
        eos_token_id=eos_token_id,
    )
    uids = rollout_batch.non_tensor_batch.get("uid")
    trajectory_ids = rollout_batch.non_tensor_batch.get("trajectory_id")
    if uids is None or len(uids) != len(rollout_batch):
        raise ValueError("boundary_return requires row-aligned uid")
    if trajectory_ids is None or len(trajectory_ids) != len(rollout_batch):
        raise ValueError("boundary_return requires row-aligned trajectory_id")

    seen_trajectories: set[tuple[str, str]] = set()
    requests: list[BoundaryContinuationRequest] = []
    for row, active in enumerate(hit_cap.tolist()):
        uid = str(uids[row])
        trajectory_id = str(trajectory_ids[row])
        identity = (uid, trajectory_id)
        if identity in seen_trajectories:
            raise ValueError(f"duplicate trajectory identity {identity!r}")
        seen_trajectories.add(identity)
        if not active:
            continue
        prompt = tuple(int(token) for token in rollout_batch.batch["prompts"][row][prompt_mask[row]].tolist())
        prefix = tuple(int(token) for token in response_token_ids[row])
        remaining = int(long_response_length) - len(prefix)
        if remaining <= 0:
            raise ValueError("cap-hit prefix leaves no positive long continuation budget")
        if len(prompt) + int(long_response_length) > int(max_model_len):
            raise ValueError(
                f"boundary_return context shortage: prompt_len={len(prompt)} + "
                f"long_response_length={long_response_length} exceeds max_model_len={max_model_len}"
            )
        requests.append(
            BoundaryContinuationRequest(
                parent_index=row,
                request_id=_request_id(policy_version, uid, trajectory_id),
                routing_key=_routing_key(policy_version, uid),
                policy_version=int(policy_version),
                uid=uid,
                trajectory_id=trajectory_id,
                prompt_token_ids=prompt,
                prefix_token_ids=prefix,
                input_token_ids=(*prompt, *prefix),
                max_tokens=remaining,
                seed=derive_continuation_seed(base_seed, policy_version, uid, trajectory_id),
            )
        )
    return requests, hit_cap


def aggregate_continuation_results(
    requests: Sequence[BoundaryContinuationRequest],
    results: Sequence[BoundaryContinuationBranchResult],
    *,
    expected_policy_version: int,
) -> tuple[BoundaryContinuationGeneration, ...]:
    """Map request/branch identities exactly and fail on missing, duplicate, or stale output."""
    request_by_id = {request.request_id: request for request in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("duplicate continuation request IDs")
    versions = {
        int(result.actual_policy_version)
        for result in results
        if result.actual_policy_version is not None
    }
    if len(versions) > 1:
        raise ValueError("continuation results have mixed actual policy versions")
    by_request: dict[str, dict[int, BoundaryContinuationBranchResult]] = {}
    for result in results:
        if result.request_id not in request_by_id:
            raise ValueError(f"unknown continuation request ID {result.request_id!r}")
        if result.actual_policy_version is None:
            raise ValueError(f"continuation request {result.request_id} is missing actual policy version")
        branches = by_request.setdefault(result.request_id, {})
        if int(result.branch_id) in branches:
            raise ValueError(f"duplicate continuation branch {result.branch_id} for {result.request_id}")
        branches[int(result.branch_id)] = result

    generations: list[BoundaryContinuationGeneration] = []
    for request in requests:
        if int(request.policy_version) != int(expected_policy_version):
            raise ValueError("continuation requests have stale or mixed requested policy versions")
        branches = by_request.get(request.request_id, {})
        if not branches:
            raise ValueError(f"missing continuation branch for {request.request_id}")
        if set(branches) != {0}:
            raise ValueError(f"continuation request {request.request_id} returned invalid branch IDs")
        result = branches[0]
        if int(result.actual_policy_version) != int(expected_policy_version):
            raise ValueError(
                f"continuation actual policy version {result.actual_policy_version} does not match "
                f"published version {expected_policy_version}"
            )
        generations.append(
            BoundaryContinuationGeneration(
                parent_index=request.parent_index,
                request_id=request.request_id,
                branch_id=0,
                uid=request.uid,
                trajectory_id=request.trajectory_id,
                prompt_token_ids=request.prompt_token_ids,
                prefix_token_ids=request.prefix_token_ids,
                tail_token_ids=tuple(int(token) for token in result.tail_token_ids),
                actual_policy_version=int(result.actual_policy_version),
            )
        )
    return tuple(generations)


@auto_await
async def run_boundary_continuations(
    *,
    config: Any,
    rollout_batch: DataProto,
    client: Any,
    eos_token_id: int | None,
    short_response_length: int,
    max_model_len: int,
    policy_version: int,
    sampling_params: Mapping[str, Any],
) -> BoundaryContinuationCapture | None:
    """Generate natural K=1 tails; the off path is inert before cap detection."""
    if config.mode == "off":
        return None
    config.validate()
    requests, hit_cap = build_continuation_requests(
        rollout_batch,
        policy_version=policy_version,
        short_response_length=short_response_length,
        long_response_length=config.long_response_length,
        max_model_len=max_model_len,
        eos_token_id=eos_token_id,
        base_seed=config.seed,
    )
    prompt_width = rollout_batch.batch["prompts"].shape[-1]
    normal_response_tokens = int(
        rollout_batch.batch["attention_mask"][:, prompt_width:].sum().item()
    )
    if not requests:
        return BoundaryContinuationCapture(
            hit_response_cap=hit_cap,
            requests=(),
            generations=(),
            normal_response_tokens=normal_response_tokens,
        )

    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def generate_one(request: BoundaryContinuationRequest) -> list[BoundaryContinuationBranchResult]:
        params = dict(sampling_params)
        params.update({"n": 1, "seed": request.seed, "max_tokens": request.max_tokens})
        async with semaphore:
            outputs = await client.generate_grouped(
                request.request_id,
                prompt_ids=list(request.input_token_ids),
                sampling_params=params,
                routing_key=request.routing_key,
            )
        results: list[BoundaryContinuationBranchResult] = []
        for fallback_branch, output in enumerate(outputs):
            extra_fields = output.extra_fields or {}
            branch_id = int(extra_fields.get("branch_id", fallback_branch))
            actual_version = extra_fields.get("global_steps")
            results.append(
                BoundaryContinuationBranchResult(
                    request_id=request.request_id,
                    branch_id=branch_id,
                    tail_token_ids=tuple(int(token) for token in output.token_ids),
                    actual_policy_version=int(actual_version) if actual_version is not None else None,
                )
            )
        return results

    results: list[BoundaryContinuationBranchResult] = []
    for start in range(0, len(requests), config.request_batch_size):
        chunk = requests[start : start + config.request_batch_size]
        tasks = [asyncio.create_task(generate_one(request)) for request in chunk]
        try:
            grouped = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        results.extend(result for group in grouped for result in group)
    generations = aggregate_continuation_results(
        requests,
        results,
        expected_policy_version=policy_version,
    )
    return BoundaryContinuationCapture(
        hit_response_cap=hit_cap,
        requests=tuple(requests),
        generations=generations,
        normal_response_tokens=normal_response_tokens,
    )
