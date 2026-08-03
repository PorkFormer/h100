"""Build exact full-rollout DataProto batches from frozen audit artifacts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _tokens(value: Any, name: str, *, allow_empty: bool) -> list[int]:
    if not isinstance(value, (list, tuple)) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a{' nonempty' if not allow_empty else ''} token list")
    tokens: list[int] = []
    for token in value:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError(f"{name} must contain nonnegative integer token IDs")
        tokens.append(token)
    return tokens


def _required(source: Mapping[str, Any], name: str) -> Any:
    if name not in source:
        raise ValueError(f"artifact row is missing required field {name}")
    return source[name]


def build_full_rollout_batch_from_artifacts(
    *,
    prompt_rows: list[dict],
    rollout_rows: list[dict],
    tokenizer: Any,
    pad_token_id: int,
) -> DataProto:
    """Join frozen rows and preserve response token IDs without decode/re-encode."""
    del tokenizer  # The exact-token contract intentionally performs no tokenization.
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    if not prompt_rows or not rollout_rows:
        raise ValueError("prompt_rows and rollout_rows must be nonempty")
    prompts_by_id: dict[int, dict[str, Any]] = {}
    for source in prompt_rows:
        row = dict(source)
        prompt_id = _nonnegative_int(_required(row, "prompt_id"), "prompt_id")
        if prompt_id in prompts_by_id:
            raise ValueError(f"duplicate prompt identity {prompt_id}")
        prompts_by_id[prompt_id] = row

    joined: list[tuple[tuple[str, int, int], dict[str, Any], dict[str, Any]]] = []
    identities: set[tuple[str, int, int]] = set()
    for source in rollout_rows:
        rollout = dict(source)
        model_id = _required(rollout, "model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a nonempty string")
        prompt_id = _nonnegative_int(_required(rollout, "prompt_id"), "prompt_id")
        rollout_index = _nonnegative_int(
            _required(rollout, "rollout_index"),
            "rollout_index",
        )
        identity = (model_id, prompt_id, rollout_index)
        if identity in identities:
            raise ValueError(f"duplicate rollout identity {identity}")
        identities.add(identity)
        if prompt_id not in prompts_by_id:
            raise ValueError(f"rollout identity references unknown prompt {prompt_id}")
        prompt = prompts_by_id[prompt_id]
        if _required(rollout, "prompt_hash") != _required(prompt, "prompt_hash"):
            raise ValueError(f"rollout {identity} prompt_hash mismatch")
        prompt_tokens = _tokens(
            _required(prompt, "prompt_token_ids"),
            "prompt_token_ids",
            allow_empty=False,
        )
        if "prompt_token_ids" in rollout and _tokens(
            rollout["prompt_token_ids"],
            "rollout prompt_token_ids",
            allow_empty=False,
        ) != prompt_tokens:
            raise ValueError(f"rollout {identity} prompt_token_ids mismatch")
        _required(rollout, "response_hash")
        _nonnegative_int(_required(rollout, "sampling_seed"), "sampling_seed")
        _tokens(
            _required(rollout, "response_token_ids"),
            "response_token_ids",
            allow_empty=True,
        )
        joined.append((identity, prompt, rollout))
    joined.sort(key=lambda item: item[0])

    prompt_tokens_by_row = [
        _tokens(item[1]["prompt_token_ids"], "prompt_token_ids", allow_empty=False)
        for item in joined
    ]
    response_tokens_by_row = [
        _tokens(item[2]["response_token_ids"], "response_token_ids", allow_empty=True)
        for item in joined
    ]
    prompt_width = max(map(len, prompt_tokens_by_row))
    response_width = max(map(len, response_tokens_by_row))
    if response_width == 0:
        raise ValueError("at least one artifact response must contain a token")
    prompt_tensors: list[list[int]] = []
    response_tensors: list[list[int]] = []
    prompt_masks: list[list[int]] = []
    response_masks: list[list[int]] = []
    for prompt_tokens, response_tokens in zip(
        prompt_tokens_by_row,
        response_tokens_by_row,
        strict=True,
    ):
        prompt_padding = prompt_width - len(prompt_tokens)
        response_padding = response_width - len(response_tokens)
        prompt_tensors.append([pad_token_id] * prompt_padding + prompt_tokens)
        prompt_masks.append([0] * prompt_padding + [1] * len(prompt_tokens))
        response_tensors.append(response_tokens + [pad_token_id] * response_padding)
        response_masks.append([1] * len(response_tokens) + [0] * response_padding)
    prompts = torch.tensor(prompt_tensors, dtype=torch.long)
    responses = torch.tensor(response_tensors, dtype=torch.long)
    response_mask = torch.tensor(response_masks, dtype=torch.long)
    attention_mask = torch.tensor(
        [left + right for left, right in zip(prompt_masks, response_masks, strict=True)],
        dtype=torch.long,
    )
    input_ids = torch.cat((prompts, responses), dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)

    non_tensors: dict[str, list[Any]] = {
        name: []
        for name in (
            "uid",
            "trajectory_id",
            "raw_prompt",
            "data_source",
            "reward_model",
            "extra_info",
            "model_id",
            "prompt_id",
            "rollout_index",
            "prompt_hash",
            "response_hash",
            "sampling_seed",
            "response_token_count",
        )
    }
    for identity, prompt, rollout in joined:
        model_id, prompt_id, rollout_index = identity
        merged = dict(prompt) | dict(rollout)
        raw_prompt = _required(merged, "raw_prompt")
        data_source = _required(merged, "data_source")
        extra_info = _required(merged, "extra_info")
        if "reward_model" in merged:
            reward_model = merged["reward_model"]
            if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
                raise ValueError("reward_model must contain ground_truth")
            reward_model = dict(reward_model)
        elif "ground_truth" in merged:
            reward_model = {"ground_truth": merged["ground_truth"]}
        else:
            raise ValueError("artifact row requires ground_truth or reward_model.ground_truth")
        response_count = len(
            _tokens(rollout["response_token_ids"], "response_token_ids", allow_empty=True)
        )
        if "response_token_count" in rollout and _nonnegative_int(
            rollout["response_token_count"],
            "response_token_count",
        ) != response_count:
            raise ValueError("response_token_count does not match response_token_ids")
        trajectory_id = f"{model_id}:{prompt_id}:{rollout_index}"
        values = {
            "uid": str(prompt_id),
            "trajectory_id": trajectory_id,
            "raw_prompt": raw_prompt,
            "data_source": data_source,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "model_id": model_id,
            "prompt_id": prompt_id,
            "rollout_index": rollout_index,
            "prompt_hash": rollout["prompt_hash"],
            "response_hash": rollout["response_hash"],
            "sampling_seed": rollout["sampling_seed"],
            "response_token_count": response_count,
        }
        for name, value in values.items():
            non_tensors[name].append(copy.deepcopy(value))
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        non_tensors={
            name: np.asarray(values, dtype=object)
            for name, values in non_tensors.items()
        },
    )
