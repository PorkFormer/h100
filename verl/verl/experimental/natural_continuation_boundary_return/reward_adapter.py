"""Reward-only long-response construction and boundary-return replacement."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.natural_continuation_boundary_return.runtime import (
    BoundaryContinuationCapture,
    BoundaryContinuationGeneration,
)


@dataclass(frozen=True)
class BoundaryRewardOutput:
    reward_tensor: torch.Tensor
    extra_info: dict[str, Any]


@dataclass(frozen=True)
class BoundaryRewardScalars:
    correctness: np.ndarray
    task_score: np.ndarray


@dataclass(frozen=True)
class BoundaryReturnBatchResult:
    hit_response_cap: np.ndarray
    short_acc: np.ndarray
    long_acc: np.ndarray
    boundary_acc: np.ndarray
    short_task_score: np.ndarray
    long_task_score: np.ndarray
    boundary_task_score: np.ndarray
    task_score_delta: np.ndarray
    tail_token_lengths: np.ndarray
    normal_response_tokens: int
    uids: np.ndarray
    metrics: dict[str, float]


def _boundary_internal_key(key: str) -> bool:
    return key.startswith("boundary_") or key.startswith("__boundary_")


def build_long_reward_batch(
    original_batch: DataProto,
    generations: Sequence[BoundaryContinuationGeneration],
    *,
    pad_token_id: int,
) -> DataProto:
    """Build an independent reward-only batch with original prompts and prefix+tail responses."""
    if not generations:
        raise ValueError("boundary_return long reward batch requires generations")
    parent_indices = [int(generation.parent_index) for generation in generations]
    if any(parent < 0 or parent >= len(original_batch) for parent in parent_indices):
        raise ValueError("boundary_return generation has invalid parent index")
    if len(set(parent_indices)) != len(parent_indices):
        raise ValueError("boundary_return generations have duplicate parent indices")

    prompt_width = max(len(generation.prompt_token_ids) for generation in generations)
    full_responses = [(*generation.prefix_token_ids, *generation.tail_token_ids) for generation in generations]
    response_width = max(len(response) for response in full_responses)
    if prompt_width <= 0 or response_width <= 0:
        raise ValueError("boundary_return reward rows require nonempty prompts and responses")
    device = original_batch.batch["responses"].device
    prompts = torch.full(
        (len(generations), prompt_width), int(pad_token_id), dtype=torch.long, device=device
    )
    responses = torch.full(
        (len(generations), response_width), int(pad_token_id), dtype=torch.long, device=device
    )
    prompt_mask = torch.zeros_like(prompts)
    response_mask = torch.zeros_like(responses)
    for row, (generation, full_response) in enumerate(zip(generations, full_responses, strict=True)):
        prompt = torch.tensor(generation.prompt_token_ids, dtype=torch.long, device=device)
        response = torch.tensor(full_response, dtype=torch.long, device=device)
        prompts[row, -len(prompt) :] = prompt
        prompt_mask[row, -len(prompt) :] = 1
        responses[row, : len(response)] = response
        response_mask[row, : len(response)] = 1
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    input_ids = torch.cat((prompts, responses), dim=-1)
    position_ids = torch.clamp(attention_mask.cumsum(-1) - 1, min=0)
    tensor_batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=len(generations),
    )

    short_reward_keys = set(original_batch.meta_info.get("reward_extra_keys", ()))
    non_tensor_batch: dict[str, np.ndarray] = {}
    for key, raw_values in original_batch.non_tensor_batch.items():
        if key in short_reward_keys or _boundary_internal_key(str(key)):
            continue
        values = np.asarray(raw_values)
        selected = [copy.deepcopy(values[parent]) for parent in parent_indices]
        non_tensor_batch[key] = np.asarray(selected, dtype=values.dtype)
    meta_info = {
        key: copy.deepcopy(value)
        for key, value in original_batch.meta_info.items()
        if key != "reward_extra_keys" and not _boundary_internal_key(str(key))
    }
    return DataProto(batch=tensor_batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def _finite_scalar_array(value: Any, *, expected_count: int, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or len(array) != expected_count:
        raise ValueError(f"reward extra-info field {field_name!r} must have exactly {expected_count} rows")
    normalized: list[float] = []
    for row, item in enumerate(array.tolist()):
        if isinstance(item, np.ndarray) and item.ndim == 0:
            item = item.item()
        if isinstance(item, bool):
            number = float(item)
        elif isinstance(item, Real):
            number = float(item)
        else:
            raise ValueError(f"reward extra-info field {field_name!r} row {row} must be a finite scalar")
        if not math.isfinite(number):
            raise ValueError(f"reward extra-info field {field_name!r} must be finite")
        normalized.append(number)
    return np.asarray(normalized, dtype=np.float64)


def extract_required_reward_scalars(
    reward_output: BoundaryRewardOutput,
    *,
    expected_count: int,
    correctness_key: str,
    task_score_key: str,
) -> BoundaryRewardScalars:
    """Extract explicit aligned correctness and task score; shaped reward is never a fallback."""
    if correctness_key not in reward_output.extra_info:
        raise ValueError(f"reward output is missing required correctness field {correctness_key!r}")
    if task_score_key not in reward_output.extra_info:
        raise ValueError(f"reward output is missing required task-score field {task_score_key!r}")
    for failure_key in ("error", "timeout"):
        if failure_key not in reward_output.extra_info:
            continue
        failures = np.asarray(reward_output.extra_info[failure_key], dtype=object).reshape(-1)
        if any(value is not None and value != "" and value is not False for value in failures):
            raise ValueError(f"reward output contains verifier {failure_key}")
    return BoundaryRewardScalars(
        correctness=_finite_scalar_array(
            reward_output.extra_info[correctness_key],
            expected_count=expected_count,
            field_name=correctness_key,
        ),
        task_score=_finite_scalar_array(
            reward_output.extra_info[task_score_key],
            expected_count=expected_count,
            field_name=task_score_key,
        ),
    )


def _transition_metrics(
    short_acc: np.ndarray,
    long_acc: np.ndarray,
    hit_cap: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    short_success = short_acc >= threshold
    long_success = long_acc >= threshold
    valid = hit_cap & np.isfinite(long_acc)
    cases = {
        "h_wrong_l_wrong": valid & ~short_success & ~long_success,
        "h_wrong_l_correct": valid & ~short_success & long_success,
        "h_correct_l_correct": valid & short_success & long_success,
        "h_correct_l_wrong": valid & short_success & ~long_success,
    }
    metrics = {
        f"boundary_return/transition_{name}_count": float(mask.sum()) for name, mask in cases.items()
    }
    valid_count = int(valid.sum())
    for name, mask in cases.items():
        metrics[f"boundary_return/transition_{name}_rate"] = (
            float(mask.sum() / valid_count) if valid_count else 0.0
        )
    return metrics


def apply_boundary_return(
    candidate: DataProto,
    *,
    capture: BoundaryContinuationCapture,
    long_reward_output: BoundaryRewardOutput,
    config: Any,
) -> BoundaryReturnBatchResult:
    """Compute shadow diagnostics or exact task-return replacement at the prefix terminal token."""
    if config.mode not in {"shadow", "replace"}:
        raise ValueError("apply_boundary_return requires shadow or replace mode")
    row_count = len(candidate)
    hit_cap = np.asarray(capture.hit_response_cap, dtype=bool)
    if hit_cap.ndim != 1 or len(hit_cap) != row_count:
        raise ValueError("hit_response_cap must align with candidate rows")
    if int(hit_cap.sum()) != len(capture.generations):
        raise ValueError("every cap-hit trajectory must have exactly one long score")
    parents = [int(generation.parent_index) for generation in capture.generations]
    if len(set(parents)) != len(parents):
        raise ValueError("duplicate long score parent")
    if set(parents) != set(np.flatnonzero(hit_cap).tolist()):
        raise ValueError("every cap-hit trajectory must have exactly one long score")

    short_output = BoundaryRewardOutput(
        reward_tensor=candidate.batch["token_level_scores"],
        extra_info={
            key: candidate.non_tensor_batch[key]
            for key in (config.correctness_key, config.task_score_key)
            if key in candidate.non_tensor_batch
        },
    )
    short = extract_required_reward_scalars(
        short_output,
        expected_count=row_count,
        correctness_key=config.correctness_key,
        task_score_key=config.task_score_key,
    )
    long = extract_required_reward_scalars(
        long_reward_output,
        expected_count=len(capture.generations),
        correctness_key=config.correctness_key,
        task_score_key=config.task_score_key,
    )
    long_acc = np.full(row_count, np.nan, dtype=np.float64)
    long_task = np.full(row_count, np.nan, dtype=np.float64)
    tail_lengths = np.zeros(row_count, dtype=np.int64)
    for long_row, generation in enumerate(capture.generations):
        parent = int(generation.parent_index)
        long_acc[parent] = long.correctness[long_row]
        long_task[parent] = long.task_score[long_row]
        tail_lengths[parent] = len(generation.tail_token_ids)

    boundary_acc = short.correctness.copy()
    boundary_task = short.task_score.copy()
    boundary_acc[hit_cap] = long_acc[hit_cap]
    boundary_task[hit_cap] = long_task[hit_cap]
    task_delta = boundary_task - short.task_score

    prefix_drift_max = 0.0
    if config.mode == "replace":
        scores = candidate.batch["token_level_scores"]
        rewards = candidate.batch["token_level_rewards"]
        response_mask = candidate.batch.get("response_mask")
        if scores.ndim != 2 or rewards.shape != scores.shape or response_mask is None or response_mask.shape != scores.shape:
            raise ValueError("boundary_return requires aligned token scores, rewards, and response_mask")
        corrected = scores.clone()
        short_task_tensor = torch.as_tensor(
            short.task_score, dtype=scores.dtype, device=scores.device
        )
        for parent in np.flatnonzero(hit_cap).tolist():
            valid_positions = torch.nonzero(response_mask[parent].bool(), as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                raise ValueError(f"cap-hit row {parent} has an empty response prefix mask")
            last = int(valid_positions[-1].item())
            corrected[parent, last] = corrected[parent, last] + torch.as_tensor(
                task_delta[parent], dtype=corrected.dtype, device=corrected.device
            )
        boundary_task_tensor = torch.as_tensor(
            boundary_task, dtype=scores.dtype, device=scores.device
        )
        residual_before = scores.detach().sum(-1) - short_task_tensor
        residual_after = corrected.detach().sum(-1) - boundary_task_tensor
        prefix_drift_max = float((residual_after - residual_before).abs().max().item())
        if prefix_drift_max > 1.0e-6:
            raise ValueError(f"boundary_return prefix shaping residual drifted by {prefix_drift_max}")
        if prefix_drift_max < 1.0e-12:
            prefix_drift_max = 0.0
        candidate.batch["token_level_scores"] = corrected
        candidate.batch["token_level_rewards"] = corrected.clone()
        candidate.non_tensor_batch["boundary_acc"] = boundary_acc.copy()
        candidate.non_tensor_batch["boundary_task_score"] = boundary_task.copy()

    threshold = float(config.correctness_threshold)
    short_success = short.correctness >= threshold
    long_success = long_acc >= threshold
    valid_long = hit_cap & np.isfinite(long_acc)
    cap_failure = valid_long & ~short_success
    cap_success = valid_long & short_success
    recovered = cap_failure & long_success
    regressed = cap_success & ~long_success
    tail_values = tail_lengths[hit_cap]
    metrics = {
        "boundary_return/hit_cap_count": float(hit_cap.sum()),
        "boundary_return/hit_cap_rate": float(hit_cap.mean()) if row_count else 0.0,
        "boundary_return/valid_long_score_count": float(valid_long.sum()),
        "boundary_return/long_success_rate_given_cap": (
            float(long_success[valid_long].mean()) if valid_long.any() else 0.0
        ),
        "boundary_return/recovered_count": float(recovered.sum()),
        "boundary_return/recovered_rate_given_cap_failure": (
            float(recovered.sum() / cap_failure.sum()) if cap_failure.any() else 0.0
        ),
        "boundary_return/regressed_count": float(regressed.sum()),
        "boundary_return/regressed_rate_given_cap_success": (
            float(regressed.sum() / cap_success.sum()) if cap_success.any() else 0.0
        ),
        "boundary_return/extra_generated_tokens": float(tail_values.sum()),
        "boundary_return/extra_generated_token_ratio": (
            float(tail_values.sum() / capture.normal_response_tokens)
            if capture.normal_response_tokens
            else 0.0
        ),
        "boundary_return/tail_tokens_mean": float(tail_values.mean()) if len(tail_values) else 0.0,
        "boundary_return/tail_tokens_p50": float(np.percentile(tail_values, 50)) if len(tail_values) else 0.0,
        "boundary_return/tail_tokens_p90": float(np.percentile(tail_values, 90)) if len(tail_values) else 0.0,
        "boundary_return/task_score_delta_mean_given_cap": (
            float(task_delta[hit_cap].mean()) if hit_cap.any() else 0.0
        ),
        "boundary_return/prefix_penalty_drift_max": prefix_drift_max,
    }
    metrics.update(_transition_metrics(short.correctness, long_acc, hit_cap, threshold))
    uids = candidate.non_tensor_batch.get("uid")
    if uids is None or len(uids) != row_count:
        raise ValueError("boundary_return requires row-aligned uid for diagnostics")
    return BoundaryReturnBatchResult(
        hit_response_cap=hit_cap.copy(),
        short_acc=short.correctness.copy(),
        long_acc=long_acc,
        boundary_acc=boundary_acc,
        short_task_score=short.task_score.copy(),
        long_task_score=long_task,
        boundary_task_score=boundary_task,
        task_score_delta=task_delta,
        tail_token_lengths=tail_lengths,
        normal_response_tokens=int(capture.normal_response_tokens),
        uids=np.asarray(uids, dtype=object).copy(),
        metrics=metrics,
    )
