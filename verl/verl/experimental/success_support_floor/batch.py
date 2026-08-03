"""Text-only witness batch construction for BSSF actor and shadow forwards."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

ACTOR_BATCH_KEYS = (
    "prompts",
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "response_mask",
    "old_log_probs",
    "advantages",
)


def build_support_batch(
    *,
    prompt_tokens_by_key: Mapping[str, Sequence[int]],
    witnesses: Sequence[Mapping[str, object]],
    prompt_width: int,
    response_width: int,
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> DataProto:
    """Pad sampled cache witnesses exactly like pure-text RL actor rows."""
    if not witnesses:
        raise ValueError("support witness batch must be nonempty")
    if prompt_width <= 0 or response_width <= 0:
        raise ValueError("prompt and response widths must be positive")
    count = len(witnesses)
    prompts = torch.full((count, prompt_width), pad_token_id, dtype=torch.long, device=device)
    responses = torch.full((count, response_width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((count, prompt_width + response_width), dtype=torch.long, device=device)
    response_mask = torch.zeros((count, response_width), dtype=torch.long, device=device)
    reference = torch.empty(count, dtype=torch.float32, device=device)
    for row, witness in enumerate(witnesses):
        key = str(witness["prompt_key"])
        if key not in prompt_tokens_by_key:
            raise ValueError(f"sampled witness references unknown prompt_key {key}")
        prompt = [int(token) for token in prompt_tokens_by_key[key]]
        response = [int(token) for token in witness["response_token_ids"]]
        if not prompt or len(prompt) > prompt_width:
            raise ValueError(f"prompt length {len(prompt)} does not fit width {prompt_width}")
        if not response or len(response) > response_width:
            raise ValueError(f"response length {len(response)} does not fit width {response_width}")
        prompts[row, prompt_width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
        responses[row, : len(response)] = torch.tensor(response, dtype=torch.long, device=device)
        attention_mask[row, prompt_width - len(prompt) : prompt_width + len(response)] = 1
        response_mask[row, : len(response)] = 1
        reference[row] = float(witness["reference_seq_logprob"])
    input_ids = torch.cat((prompts, responses), dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)
    zeros = torch.zeros((count, response_width), dtype=torch.float32, device=device)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "old_log_probs": zeros.clone(),
            "advantages": zeros.clone(),
            "support_ref_seq_logprob": reference,
        }
    )


def build_augmented_actor_batch(
    rl_batch: DataProto,
    support_batch: DataProto,
    *,
    lambda_value: float,
    alpha: float,
    global_support_batch_size: int,
) -> DataProto:
    """Return a new actor-only batch; never mutate training statistics in ``rl_batch``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if lambda_value < 0.0 or not math.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite and nonnegative")
    if global_support_batch_size <= 0:
        raise ValueError("global_support_batch_size must be positive")
    missing = [key for key in ACTOR_BATCH_KEYS if key not in rl_batch.batch]
    if missing:
        raise ValueError(f"RL actor batch is missing required fields {missing}")
    if rl_batch.batch["prompts"].shape[1:] != support_batch.batch["prompts"].shape[1:]:
        raise ValueError("RL and support prompt widths must match")
    if rl_batch.batch["responses"].shape[1:] != support_batch.batch["responses"].shape[1:]:
        raise ValueError("RL and support response widths must match")

    rl = rl_batch.select(batch_keys=list(ACTOR_BATCH_KEYS), non_tensor_batch_keys=[])
    rl = DataProto.from_dict(tensors={key: value.clone() for key, value in rl.batch.items()})
    rl_count = len(rl)
    response_mask = rl.batch["response_mask"]
    rl.batch["ppo_response_mask"] = response_mask.clone()
    rl.batch["support_response_mask"] = torch.zeros_like(response_mask)
    rl.batch["support_sample_mask"] = torch.zeros(rl_count, dtype=torch.bool, device=response_mask.device)
    rl.batch["support_ref_seq_logprob"] = torch.zeros(rl_count, dtype=torch.float32, device=response_mask.device)

    support = support_batch.select(
        batch_keys=list(ACTOR_BATCH_KEYS) + ["support_ref_seq_logprob"],
        non_tensor_batch_keys=[],
    )
    support = DataProto.from_dict(tensors={key: value.clone() for key, value in support.batch.items()})
    support_count = len(support)
    support_response_mask = support.batch["response_mask"]
    support.batch["ppo_response_mask"] = torch.zeros_like(support_response_mask)
    support.batch["support_response_mask"] = support_response_mask.clone()
    support.batch["support_sample_mask"] = torch.ones(
        support_count, dtype=torch.bool, device=support_response_mask.device
    )
    augmented = DataProto.concat([rl, support])
    device = augmented.batch.device
    count = len(augmented)
    augmented.batch["support_lambda"] = torch.full(
        (count,), lambda_value, dtype=torch.float32, device=device
    )
    augmented.batch["support_log_alpha"] = torch.full(
        (count,), math.log(alpha), dtype=torch.float32, device=device
    )
    augmented.batch["support_global_batch_size"] = torch.full(
        (count,), global_support_batch_size, dtype=torch.long, device=device
    )
    return augmented
