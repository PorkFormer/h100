"""Auxiliary forced-answer diagnostics for response-cap trajectories.

Probe generations remain independent inference requests. Nothing in this module
adds probe tokens or rewards to the actor-training ``DataProto``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.ray_utils import auto_await


_LENGTH_REASONS = {"length", "max_length", "max_tokens", "token_limit"}
_EOS_REASONS = {"stop", "eos", "eos_token", "end_turn"}
_ABORT_REASONS = {"abort", "aborted", "cancelled", "canceled", "error"}


@dataclass(frozen=True)
class ForcedAnswerRequest:
    parent_index: int
    request_id: str
    routing_key: str
    prompt_token_ids: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class ForcedAnswerGeneration:
    parent_index: int
    branch_id: int
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ForcedAnswerProbeCapture:
    hit_response_cap: np.ndarray
    probe_attempted: np.ndarray
    context_overflow: np.ndarray
    generations: tuple[ForcedAnswerGeneration, ...]
    probe_input_tokens: int


@dataclass(frozen=True)
class ForcedAnswerProbeDiagnostics:
    metrics: dict[str, float]
    rewards_by_parent: dict[int, tuple[float, ...]]
    successes_by_parent: dict[int, tuple[bool, ...]]


def _normalize_finish_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    value = getattr(reason, "value", reason)
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    normalized = str(value).strip().lower()
    return normalized or None


def detect_hit_response_cap(
    *,
    finish_reasons: Sequence[Any] | None,
    response_lengths: Sequence[int],
    max_response_length: int,
    response_token_ids: Sequence[Sequence[int]] | None = None,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Distinguish explicit length termination from natural EOS.

    Backend finish metadata takes priority. When it is unavailable or ambiguous,
    a full response without an EOS token is treated as a cap hit.
    """
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    lengths = [int(length) for length in response_lengths]
    if finish_reasons is not None and len(finish_reasons) != len(lengths):
        raise ValueError("finish_reasons and response_lengths must have equal length")
    if response_token_ids is not None and len(response_token_ids) != len(lengths):
        raise ValueError("response_token_ids and response_lengths must have equal length")

    result: list[bool] = []
    for index, length in enumerate(lengths):
        reason = _normalize_finish_reason(finish_reasons[index]) if finish_reasons is not None else None
        if reason in _LENGTH_REASONS:
            result.append(True)
            continue
        if reason in _EOS_REASONS or reason in _ABORT_REASONS:
            result.append(False)
            continue

        has_eos = False
        if response_token_ids is not None and eos_token_id is not None:
            has_eos = int(eos_token_id) in [int(token_id) for token_id in response_token_ids[index][:length]]
        result.append(length >= max_response_length and not has_eos)
    return np.asarray(result, dtype=bool)


def _derive_seed(base_seed: int, global_step: int, parent_index: int, routing_key: str) -> int:
    payload = json.dumps(
        ["forced-answer", int(base_seed), int(global_step), int(parent_index), routing_key],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def build_forced_answer_requests(
    batch: DataProto,
    *,
    hit_response_cap: Sequence[bool],
    instruction_token_ids: Sequence[int],
    global_step: int,
    base_seed: int,
) -> list[ForcedAnswerRequest]:
    """Build one grouped request for each capped trajectory."""
    if len(hit_response_cap) != len(batch):
        raise ValueError("hit_response_cap must align with the rollout batch")
    prompt_width = batch.batch["prompts"].shape[-1]
    prompt_masks = batch.batch["attention_mask"][:, :prompt_width].bool()
    response_masks = batch.batch["attention_mask"][:, prompt_width:].bool()
    requests: list[ForcedAnswerRequest] = []
    uids = batch.non_tensor_batch.get("uid")
    for row, hit_cap in enumerate(hit_response_cap):
        if not bool(hit_cap):
            continue
        prompt_ids = batch.batch["prompts"][row][prompt_masks[row]].tolist()
        response_ids = batch.batch["responses"][row][response_masks[row]].tolist()
        input_ids = tuple(int(token) for token in (*prompt_ids, *response_ids, *instruction_token_ids))
        uid = str(uids[row]) if uids is not None else str(row)
        routing_key = f"forced-answer-route-{uid}"
        request_id = f"forced-answer-{global_step}-{row}-{hashlib.sha256(uid.encode()).hexdigest()[:12]}"
        requests.append(
            ForcedAnswerRequest(
                parent_index=row,
                request_id=request_id,
                routing_key=routing_key,
                prompt_token_ids=input_ids,
                seed=_derive_seed(base_seed, global_step, row, routing_key),
            )
        )
    return requests


@auto_await
async def run_forced_answer_probe(
    *,
    config: Any,
    rollout_batch: DataProto,
    tokenizer: Any,
    client: Any,
    max_response_length: int,
    max_model_len: int,
    global_step: int,
) -> ForcedAnswerProbeCapture | None:
    """Detect cap hits and generate only their short answer branches.

    The disabled path returns before detection, tokenization, or client access.
    """
    if not bool(config.enable):
        return None
    config.validate()

    prompt_width = rollout_batch.batch["prompts"].shape[-1]
    response_masks = rollout_batch.batch["attention_mask"][:, prompt_width:].bool()
    response_lengths = response_masks.sum(-1).cpu().tolist()
    response_token_ids = [
        rollout_batch.batch["responses"][row][response_masks[row]].cpu().tolist() for row in range(len(rollout_batch))
    ]
    finish_reasons = rollout_batch.non_tensor_batch.get("finish_reason")
    hit_cap = detect_hit_response_cap(
        finish_reasons=finish_reasons,
        response_lengths=response_lengths,
        max_response_length=max_response_length,
        response_token_ids=response_token_ids,
        eos_token_id=tokenizer.eos_token_id,
    )
    encoded_instruction = tokenizer(
        config.instruction,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    requests = build_forced_answer_requests(
        rollout_batch,
        hit_response_cap=hit_cap,
        instruction_token_ids=encoded_instruction,
        global_step=global_step,
        base_seed=config.seed,
    )
    probe_attempted = np.zeros(len(rollout_batch), dtype=bool)
    context_overflow = np.zeros(len(rollout_batch), dtype=bool)
    runnable_requests: list[ForcedAnswerRequest] = []
    probe_input_tokens = 0
    for request in requests:
        required_context = len(request.prompt_token_ids) + config.max_new_tokens
        if required_context > max_model_len:
            context_overflow[request.parent_index] = True
            continue
        probe_attempted[request.parent_index] = True
        probe_input_tokens += len(request.prompt_token_ids)
        runnable_requests.append(request)
    requests = runnable_requests
    if not requests:
        return ForcedAnswerProbeCapture(
            hit_response_cap=hit_cap,
            probe_attempted=probe_attempted,
            context_overflow=context_overflow,
            generations=(),
            probe_input_tokens=0,
        )

    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def generate_one(request: ForcedAnswerRequest) -> list[ForcedAnswerGeneration]:
        sampling_params = {
            "n": config.num_samples,
            "seed": request.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": -1,
            "max_tokens": config.max_new_tokens,
        }
        async with semaphore:
            outputs = await client.generate_grouped(
                request.request_id,
                prompt_ids=list(request.prompt_token_ids),
                sampling_params=sampling_params,
                routing_key=request.routing_key,
            )
        if config.strict and len(outputs) != config.num_samples:
            raise RuntimeError(
                f"Forced-answer request {request.request_id} returned {len(outputs)} branches; "
                f"expected {config.num_samples}"
            )
        generations: list[ForcedAnswerGeneration] = []
        seen_branches: set[int] = set()
        for fallback_branch, output in enumerate(outputs):
            branch_id = int((output.extra_fields or {}).get("branch_id", fallback_branch))
            token_ids = tuple(int(token_id) for token_id in output.token_ids)
            if config.strict and (branch_id in seen_branches or not token_ids):
                raise RuntimeError(f"Malformed forced-answer branch {branch_id} for {request.request_id}")
            seen_branches.add(branch_id)
            generations.append(
                ForcedAnswerGeneration(
                    parent_index=request.parent_index,
                    branch_id=branch_id,
                    prompt_token_ids=request.prompt_token_ids,
                    response_token_ids=token_ids,
                )
            )
        if config.strict and seen_branches != set(range(config.num_samples)):
            raise RuntimeError(f"Forced-answer request {request.request_id} returned invalid branch IDs")
        return generations

    grouped = await asyncio.gather(*(generate_one(request) for request in requests))
    generations = tuple(generation for request_results in grouped for generation in request_results)
    return ForcedAnswerProbeCapture(
        hit_response_cap=hit_cap,
        probe_attempted=probe_attempted,
        context_overflow=context_overflow,
        generations=generations,
        probe_input_tokens=probe_input_tokens,
    )


def build_probe_reward_batch(
    original_batch: DataProto,
    generations: Sequence[ForcedAnswerGeneration],
    *,
    pad_token_id: int,
) -> DataProto:
    """Create a reward-only batch that is independent from actor training tensors."""
    if not generations:
        raise ValueError("generations must be nonempty")
    prompt_width = max(len(generation.prompt_token_ids) for generation in generations)
    response_width = max(len(generation.response_token_ids) for generation in generations)
    prompts = torch.full((len(generations), prompt_width), pad_token_id, dtype=torch.long)
    responses = torch.full((len(generations), response_width), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompts)
    response_mask = torch.zeros_like(responses)
    for row, generation in enumerate(generations):
        prompt = torch.tensor(generation.prompt_token_ids, dtype=torch.long)
        response = torch.tensor(generation.response_token_ids, dtype=torch.long)
        prompts[row, -len(prompt) :] = prompt
        prompt_mask[row, -len(prompt) :] = 1
        responses[row, : len(response)] = response
        response_mask[row, : len(response)] = 1
    input_ids = torch.cat((prompts, responses), dim=-1)
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    position_ids = torch.clamp(attention_mask.cumsum(-1) - 1, min=0)
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=len(generations),
    )
    parent_indices = [generation.parent_index for generation in generations]
    non_tensor_batch = {}
    for key, values in original_batch.non_tensor_batch.items():
        array = np.asarray(values)
        non_tensor_batch[key] = np.asarray([array[index] for index in parent_indices], dtype=array.dtype)
    non_tensor_batch["probe_parent_index"] = np.asarray(parent_indices, dtype=np.int64)
    non_tensor_batch["probe_branch_id"] = np.asarray(
        [generation.branch_id for generation in generations], dtype=np.int64
    )
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def aggregate_probe_diagnostics(
    *,
    hit_response_cap: Sequence[bool],
    probe_attempted: Sequence[bool],
    context_overflow: Sequence[bool],
    generations: Sequence[ForcedAnswerGeneration],
    probe_correctness: Sequence[float],
    original_correctness: Sequence[float],
    probe_shaped_rewards: Sequence[float],
    original_shaped_rewards: Sequence[float],
    original_generated_tokens: int,
    probe_input_tokens: int,
    num_samples: int,
    correctness_threshold: float,
    high_confidence_threshold: float,
) -> ForcedAnswerProbeDiagnostics:
    """Aggregate raw correctness separately from shaped-reward telemetry."""
    hit_cap = np.asarray(hit_response_cap, dtype=bool)
    attempted = np.asarray(probe_attempted, dtype=bool)
    overflow = np.asarray(context_overflow, dtype=bool)
    original_correct = np.asarray(original_correctness, dtype=np.float64)
    original_shaped = np.asarray(original_shaped_rewards, dtype=np.float64)
    probe_correct = np.asarray(probe_correctness, dtype=np.float64)
    probe_shaped = np.asarray(probe_shaped_rewards, dtype=np.float64)
    if len(attempted) != len(hit_cap):
        raise ValueError("probe_attempted must align with hit_response_cap")
    if len(overflow) != len(hit_cap):
        raise ValueError("context_overflow must align with hit_response_cap")
    if np.any(attempted & ~hit_cap) or np.any(overflow & ~hit_cap):
        raise ValueError("probe state may only be set for truncated trajectories")
    if np.any(attempted & overflow):
        raise ValueError("context-overflow trajectories cannot be attempted")
    if len(original_correct) != len(hit_cap):
        raise ValueError("original_correctness must align with hit_response_cap")
    if len(original_shaped) != len(hit_cap):
        raise ValueError("original_shaped_rewards must align with hit_response_cap")
    if len(generations) != len(probe_correct):
        raise ValueError("probe_correctness must align with generations")
    if len(generations) != len(probe_shaped):
        raise ValueError("probe_shaped_rewards must align with generations")

    shaped_by_parent: dict[int, list[tuple[int, float]]] = {}
    correctness_by_parent: dict[int, list[tuple[int, float]]] = {}
    tokens_by_parent: dict[int, int] = {}
    for generation, correctness, shaped_reward in zip(
        generations, probe_correct, probe_shaped, strict=True
    ):
        if not hit_cap[generation.parent_index]:
            raise ValueError("probe generation points to a non-truncated trajectory")
        if not attempted[generation.parent_index]:
            raise ValueError("probe generation points to an unattempted trajectory")
        shaped_by_parent.setdefault(generation.parent_index, []).append(
            (generation.branch_id, float(shaped_reward))
        )
        correctness_by_parent.setdefault(generation.parent_index, []).append(
            (generation.branch_id, float(correctness))
        )
        tokens_by_parent[generation.parent_index] = tokens_by_parent.get(generation.parent_index, 0) + len(
            generation.response_token_ids
        )

    ordered_rewards: dict[int, tuple[float, ...]] = {}
    successes: dict[int, tuple[bool, ...]] = {}
    for parent_index in np.flatnonzero(attempted).tolist():
        shaped_branches = sorted(shaped_by_parent.get(parent_index, []))
        correctness_branches = sorted(correctness_by_parent.get(parent_index, []))
        expected_branches = list(range(num_samples))
        if (
            len(shaped_branches) != num_samples
            or [branch for branch, _ in shaped_branches] != expected_branches
            or [branch for branch, _ in correctness_branches] != expected_branches
        ):
            raise ValueError(f"trajectory {parent_index} does not have exactly {num_samples} probe branches")
        parent_rewards = tuple(reward for _, reward in shaped_branches)
        parent_correctness = tuple(value for _, value in correctness_branches)
        ordered_rewards[parent_index] = parent_rewards
        successes[parent_index] = tuple(value >= correctness_threshold for value in parent_correctness)

    truncated_count = int(hit_cap.sum())
    attempted_count = int(attempted.sum())
    overflow_count = int(overflow.sum())
    total_count = len(hit_cap)
    extra_tokens = int(sum(tokens_by_parent.values()))
    success_rates = [float(np.mean(successes[parent])) for parent in sorted(successes)]
    any_success = [any(successes[parent]) for parent in sorted(successes)]
    all_success = [all(successes[parent]) for parent in sorted(successes)]
    candidate = [
        bool(original_correct[parent] < correctness_threshold and any(successes[parent]))
        for parent in sorted(successes)
    ]
    high_confidence = [
        bool(
            original_correct[parent] < correctness_threshold
            and float(np.mean(successes[parent])) >= high_confidence_threshold
        )
        for parent in sorted(successes)
    ]
    truncated_failures = [
        parent
        for parent in np.flatnonzero(hit_cap & attempted).tolist()
        if original_correct[parent] < correctness_threshold
    ]
    recovered_failures = sum(any(successes[parent]) for parent in truncated_failures)

    denominator = float(truncated_count) if truncated_count else 1.0
    extra_total_tokens = probe_input_tokens + extra_tokens
    metrics = {
        "probe/hit_cap_rate": float(truncated_count / total_count) if total_count else 0.0,
        "probe/num_truncated_trajectories": float(truncated_count),
        "probe/num_probe_generations": float(len(generations)),
        "probe/context_overflow_count": float(overflow_count),
        "probe/context_overflow_rate": float(overflow_count / denominator),
        "probe/probe_attempted_count": float(attempted_count),
        "probe/probe_coverage_rate": float(attempted_count / denominator),
        "probe/success_rate_mean": float(np.mean(success_rates)) if success_rates else 0.0,
        "probe/p_any_success": float(np.mean(any_success)) if any_success else 0.0,
        "probe/p_all_success": float(np.mean(all_success)) if all_success else 0.0,
        "probe/extra_input_tokens": float(probe_input_tokens),
        "probe/extra_generated_tokens": float(extra_tokens),
        "probe/extra_total_tokens": float(extra_total_tokens),
        "probe/extra_generated_token_ratio": (
            float(extra_tokens / original_generated_tokens) if original_generated_tokens > 0 else 0.0
        ),
        "probe/extra_total_token_ratio": (
            float(extra_total_tokens / original_generated_tokens) if original_generated_tokens > 0 else 0.0
        ),
        # Legacy dashboard alias for generated-token ratio.
        "probe/extra_token_ratio": (
            float(extra_tokens / original_generated_tokens) if original_generated_tokens > 0 else 0.0
        ),
        "probe/raw_correctness_mean": float(np.mean(probe_correct)) if len(probe_correct) else 0.0,
        "probe/shaped_reward_mean": float(np.mean(probe_shaped)) if len(probe_shaped) else 0.0,
        "probe/original_truncated_raw_correctness_mean": (
            float(np.mean(original_correct[hit_cap])) if truncated_count else 0.0
        ),
        "probe/original_truncated_shaped_reward_mean": (
            float(np.mean(original_shaped[hit_cap])) if truncated_count else 0.0
        ),
        # Backward-compatible telemetry alias; never used to determine correctness.
        "probe/reward_mean": float(np.mean(probe_shaped)) if len(probe_shaped) else 0.0,
        "probe/truncation_false_negative_candidate_rate": float(sum(candidate) / denominator),
        "probe/truncation_high_confidence_recoverable_rate": float(sum(high_confidence) / denominator),
        "probe/recovery_rate_given_truncated_failure": (
            float(recovered_failures / len(truncated_failures)) if truncated_failures else 0.0
        ),
    }
    return ForcedAnswerProbeDiagnostics(
        metrics=metrics,
        rewards_by_parent={parent: tuple(values) for parent, values in ordered_rewards.items()},
        successes_by_parent=successes,
    )


def save_probe_examples(
    *,
    output_dir: str,
    global_step: int,
    original_batch: DataProto,
    generations: Sequence[ForcedAnswerGeneration],
    diagnostics: ForcedAnswerProbeDiagnostics,
    tokenizer: Any,
    max_examples: int,
    response_tail_chars: int,
    parent_indices: Sequence[int] | None = None,
) -> Path | None:
    """Write a bounded per-step JSONL diagnostic sample."""
    if max_examples <= 0 or not diagnostics.rewards_by_parent:
        return None
    by_parent: dict[int, list[ForcedAnswerGeneration]] = {}
    for generation in generations:
        by_parent.setdefault(generation.parent_index, []).append(generation)
    destination = Path(os.path.expanduser(output_dir)).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"step_{global_step}.jsonl"
    prompt_width = original_batch.batch["prompts"].shape[-1]
    response_masks = original_batch.batch["attention_mask"][:, prompt_width:].bool()
    original_scores = original_batch.batch["rm_scores"].sum(-1).cpu().tolist()
    if parent_indices is None:
        parent_indices = list(range(len(original_batch)))
    if len(parent_indices) != len(original_batch):
        raise ValueError("parent_indices must align with original_batch")
    row_by_parent = {int(parent): row for row, parent in enumerate(parent_indices)}
    with path.open("w", encoding="utf-8") as handle:
        for parent_index in sorted(diagnostics.rewards_by_parent)[:max_examples]:
            row = row_by_parent[parent_index]
            parent_generations = sorted(by_parent[parent_index], key=lambda item: item.branch_id)
            response_ids = original_batch.batch["responses"][row][response_masks[row]]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            rewards = diagnostics.rewards_by_parent[parent_index]
            record: Mapping[str, Any] = {
                "global_step": global_step,
                "prompt_id": str(
                    original_batch.non_tensor_batch.get("uid", np.arange(len(original_batch)))[row]
                ),
                "original_response_length": int(response_masks[row].sum().item()),
                "hit_cap": True,
                "original_reward": float(original_scores[row]),
                "original_response_tail": response_text[-response_tail_chars:] if response_tail_chars else "",
                "probe_answers": [
                    tokenizer.decode(generation.response_token_ids, skip_special_tokens=True)
                    for generation in parent_generations
                ],
                "probe_rewards": list(rewards),
                "probe_success_rate": float(np.mean(diagnostics.successes_by_parent[parent_index])),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
