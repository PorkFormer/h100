# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict natural-continuation request construction and result mapping."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from verl import DataProto
from verl.trainer.ppo.forced_answer_probe import detect_hit_response_cap
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _emit_audit_event(message: str, *args: Any, level: int = logging.INFO) -> None:
    """Emit cleanup/order evidence even when Ray worker logging is not forwarded."""
    rendered = message % args
    logger.log(level, rendered)
    print(rendered, flush=True)


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
    stop_reason: str | None = None
    finish_reason: str | None = None


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
    stop_reason: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class BoundaryContinuationCapture:
    hit_response_cap: np.ndarray
    requests: tuple[BoundaryContinuationRequest, ...]
    generations: tuple[BoundaryContinuationGeneration, ...]
    normal_response_tokens: int
    continuation_input_token_lengths: tuple[int, ...] = ()
    long_hit_response_cap: np.ndarray | None = None
    request_timeout_seconds: float = 0.0
    continuation_timeout_count: int = 0


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
        def normalize_reason(reason: Any) -> str | None:
            if reason is None:
                return None
            reason = getattr(reason, "value", reason)
            return str(reason).strip().lower().split(".")[-1]

        stop_reason = normalize_reason(result.stop_reason)
        finish_reason = normalize_reason(result.finish_reason)
        invalid_terminal_markers = ("abort", "error", "cancel", "timeout")
        if (
            stop_reason is None
            or finish_reason is None
            or any(marker in stop_reason for marker in invalid_terminal_markers)
            or any(marker in finish_reason for marker in invalid_terminal_markers)
        ):
            raise ValueError(
                f"continuation request {request.request_id} has invalid terminal state "
                f"stop_reason={result.stop_reason!r}, finish_reason={result.finish_reason!r}"
            )
        if len(result.tail_token_ids) > int(request.max_tokens):
            raise ValueError(
                f"continuation request {request.request_id} returned {len(result.tail_token_ids)} tokens "
                f"above max_tokens={request.max_tokens}"
            )
        if not result.tail_token_ids and not (
            {stop_reason, finish_reason} & {"eos", "stop", "completed"}
        ):
            raise ValueError(
                f"continuation request {request.request_id} returned a zero-token tail with illegal "
                f"terminal state stop_reason={result.stop_reason!r}, finish_reason={result.finish_reason!r}"
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
                stop_reason=result.stop_reason,
                finish_reason=result.finish_reason,
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
            continuation_input_token_lengths=(),
            long_hit_response_cap=np.zeros(len(rollout_batch), dtype=bool),
            request_timeout_seconds=float(config.request_timeout_seconds),
        )

    _emit_audit_event(
        "boundary_return event=continuation_start policy_version=%d request_count=%d timeout_seconds=%s",
        policy_version,
        len(requests),
        config.request_timeout_seconds,
    )
    if not hasattr(client, "start_grouped"):
        raise TypeError("boundary_return requires a tracked grouped-request client")

    async def run_wave(
        wave: Sequence[BoundaryContinuationRequest],
    ) -> list[list[BoundaryContinuationBranchResult]]:
        tracked_requests: list[Any] = []
        completed_tracked_ids: set[int] = set()

        def request_params(request: BoundaryContinuationRequest) -> dict[str, Any]:
            params = dict(sampling_params)
            params.update({"n": 1, "seed": request.seed, "max_tokens": request.max_tokens})
            return params

        async def start_one(request: BoundaryContinuationRequest) -> tuple[Any, Any]:
            tracked = await client.start_grouped(
                request.request_id,
                prompt_ids=list(request.input_token_ids),
                sampling_params=request_params(request),
                routing_key=request.routing_key,
            )
            return request, tracked

        async def generate_one(
            request: BoundaryContinuationRequest,
            tracked: Any,
        ) -> list[BoundaryContinuationBranchResult]:
            outputs = await asyncio.wait_for(
                tracked.result(), timeout=float(config.request_timeout_seconds)
            )
            completed_tracked_ids.add(id(tracked))
            wave_results: list[BoundaryContinuationBranchResult] = []
            for fallback_branch, output in enumerate(outputs):
                extra_fields = output.extra_fields or {}
                branch_id = int(extra_fields.get("branch_id", fallback_branch))
                actual_version = extra_fields.get("global_steps")
                wave_results.append(
                    BoundaryContinuationBranchResult(
                        request_id=request.request_id,
                        branch_id=branch_id,
                        tail_token_ids=tuple(int(token) for token in output.token_ids),
                        actual_policy_version=int(actual_version) if actual_version is not None else None,
                        stop_reason=output.stop_reason,
                        finish_reason=extra_fields.get("finish_reason"),
                    )
                )
            return wave_results

        async def cleanup(primary_error: BaseException, tasks: Sequence[asyncio.Task]) -> None:
            cleanup_errors: list[BaseException] = []
            active = [tracked for tracked in tracked_requests if id(tracked) not in completed_tracked_ids]
            _emit_audit_event(
                "boundary_return cleanup event=abort_start backend_request_ids=%s",
                [tracked.backend_request_id for tracked in active],
                level=logging.WARNING,
            )
            abort_results = await asyncio.gather(
                *(tracked.abort() for tracked in active), return_exceptions=True
            )
            abort_errors = [result for result in abort_results if isinstance(result, BaseException)]
            cleanup_errors.extend(abort_errors)
            _emit_audit_event(
                "boundary_return cleanup event=abort_ack count=%d errors=%d",
                len(active),
                len(abort_errors),
                level=logging.WARNING,
            )

            drain_handles: dict[str, Any] = {}
            for tracked in tracked_requests:
                drain_handles.setdefault(str(tracked.server_id), tracked)
            _emit_audit_event(
                "boundary_return cleanup event=drain_start server_ids=%s",
                list(drain_handles),
                level=logging.WARNING,
            )
            drain_results = await asyncio.gather(
                *(tracked.drain() for tracked in drain_handles.values()), return_exceptions=True
            )
            drain_errors = [result for result in drain_results if isinstance(result, BaseException)]
            cleanup_errors.extend(drain_errors)
            _emit_audit_event(
                "boundary_return cleanup event=drain_ack count=%d errors=%d",
                len(drain_handles),
                len(drain_errors),
                level=logging.WARNING,
            )

            release_errors: list[BaseException] = []
            if not drain_errors:
                _emit_audit_event(
                    "boundary_return cleanup event=release_start server_ids=%s",
                    [tracked.server_id for tracked in tracked_requests],
                    level=logging.WARNING,
                )
                release_results = await asyncio.gather(
                    *(tracked.release() for tracked in tracked_requests), return_exceptions=True
                )
                release_errors = [
                    result for result in release_results if isinstance(result, BaseException)
                ]
                cleanup_errors.extend(release_errors)
                _emit_audit_event(
                    "boundary_return cleanup event=release_ack count=%d errors=%d",
                    len(tracked_requests),
                    len(release_errors),
                    level=logging.WARNING,
                )
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            _emit_audit_event(
                "boundary_return cleanup event=local_settle count=%d",
                len(tasks),
                level=logging.WARNING,
            )
            for cleanup_error in cleanup_errors:
                primary_error.add_note(
                    f"boundary continuation cleanup error: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            cleanup_attested = (
                getattr(primary_error, "boundary_remote_cleanup_attested", True)
                and not drain_errors
                and not release_errors
            )
            setattr(primary_error, "boundary_remote_cleanup_attested", cleanup_attested)
            setattr(
                primary_error,
                "boundary_continuation_timeout_count",
                int(isinstance(primary_error, TimeoutError)),
            )
            setattr(
                primary_error,
                "boundary_continuation_request_timeout_seconds",
                float(config.request_timeout_seconds),
            )
            _emit_audit_event(
                "boundary_return event=request_failure error_type=%s timeout_count=%d "
                "timeout_seconds=%s cleanup_attested=%s",
                type(primary_error).__name__,
                int(isinstance(primary_error, TimeoutError)),
                config.request_timeout_seconds,
                cleanup_attested,
                level=logging.WARNING,
            )

        start_results = await asyncio.gather(
            *(start_one(request) for request in wave), return_exceptions=True
        )
        primary_start_error = next(
            (result for result in start_results if isinstance(result, BaseException)), None
        )
        request_handles: list[tuple[BoundaryContinuationRequest, Any]] = []
        for result in start_results:
            if not isinstance(result, BaseException):
                request, tracked = result
                tracked_requests.append(tracked)
                request_handles.append((request, tracked))
        if primary_start_error is not None:
            await cleanup(primary_start_error, ())
            raise primary_start_error

        tasks = [
            asyncio.create_task(generate_one(request, tracked))
            for request, tracked in request_handles
        ]
        try:
            grouped = await asyncio.gather(*tasks)
            for tracked in tracked_requests:
                await tracked.release()
            return grouped
        except BaseException as primary_error:
            await cleanup(primary_error, tasks)
            raise

    results: list[BoundaryContinuationBranchResult] = []
    for start in range(0, len(requests), config.request_batch_size):
        chunk = requests[start : start + config.request_batch_size]
        for wave_start in range(0, len(chunk), config.max_concurrent_requests):
            wave = chunk[wave_start : wave_start + config.max_concurrent_requests]
            grouped = await run_wave(wave)
            results.extend(result for group in grouped for result in group)
    generations = aggregate_continuation_results(
        requests,
        results,
        expected_policy_version=policy_version,
    )
    _emit_audit_event(
        "boundary_return event=continuation_complete policy_version=%d request_count=%d",
        policy_version,
        len(requests),
    )
    requests_by_id = {request.request_id: request for request in requests}
    long_hit_response_cap = np.zeros(len(rollout_batch), dtype=bool)
    for generation in generations:
        request = requests_by_id[generation.request_id]
        long_hit_response_cap[generation.parent_index] = (
            str(generation.finish_reason).lower() == "length"
            or len(generation.tail_token_ids) >= request.max_tokens
        )
    return BoundaryContinuationCapture(
        hit_response_cap=hit_cap,
        requests=tuple(requests),
        generations=generations,
        normal_response_tokens=normal_response_tokens,
        continuation_input_token_lengths=tuple(len(request.input_token_ids) for request in requests),
        long_hit_response_cap=long_hit_response_cap,
        request_timeout_seconds=float(config.request_timeout_seconds),
        continuation_timeout_count=0,
    )
