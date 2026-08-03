"""Protected-group resolution and exact-token OBCF prefix batches."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.capability_constraints.identity import canonical_prompt_key
from verl.utils.model import compute_position_id_with_mask


@dataclass(frozen=True)
class ProtectedGroupSelection:
    rollout_indices: torch.Tensor
    group_ids: torch.Tensor
    prompt_keys: tuple[str, ...]
    capability_floors: torch.Tensor
    rollout_count_per_group: int


def _left_padded_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> list[int]:
    if tokens.ndim != 1 or mask.shape != tokens.shape:
        raise ValueError("prompt tokens and mask must be one-dimensional and aligned")
    binary = mask == 1
    if not bool(((mask == 0) | binary).all().item()) or not bool(binary.any().item()):
        raise ValueError("prompt attention mask must be binary and nonempty")
    first = int(torch.nonzero(binary, as_tuple=False)[0].item())
    if not bool(binary[first:].all().item()) or bool(binary[:first].any().item()):
        raise ValueError("prompt attention mask must be left padded")
    return [int(token) for token in tokens[first:].tolist()]


def resolve_protected_groups(
    *,
    batch: DataProto,
    cache: Any,
    rollout_n: int,
) -> ProtectedGroupSelection | None:
    """Resolve protected retained prompt occurrences in deterministic UID order."""
    if not isinstance(rollout_n, int) or isinstance(rollout_n, bool) or rollout_n <= 0:
        raise ValueError("rollout_n must be a positive integer")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("retained batch is missing uid")
    for key in ("prompts", "attention_mask"):
        if batch.batch is None or key not in batch.batch:
            raise ValueError(f"retained batch is missing {key}")
    prompt_width = batch.batch["prompts"].shape[1]
    if batch.batch["attention_mask"].shape[1] < prompt_width:
        raise ValueError("attention_mask is shorter than prompts")
    uids = [str(uid) for uid in batch.non_tensor_batch["uid"].tolist()]
    if len(uids) != len(batch) or not all(uids):
        raise ValueError("uid must contain one nonempty identity per rollout")
    rows_by_uid: dict[str, list[int]] = {}
    for row, uid in enumerate(uids):
        rows_by_uid.setdefault(uid, []).append(row)
    if any(len(rows) != rollout_n for rows in rows_by_uid.values()):
        raise ValueError("every retained uid must occur exactly rollout_n times")

    tokenizer_fp = str(cache.manifest["tokenizer_fingerprint"])
    template_fp = str(cache.manifest["chat_template_fingerprint"])
    selected_rows: list[int] = []
    selected_group_ids: list[int] = []
    selected_keys: list[str] = []
    floors: list[float] = []
    for rows in rows_by_uid.values():
        group_tokens = [
            _left_padded_tokens(
                batch.batch["prompts"][row],
                batch.batch["attention_mask"][row, :prompt_width],
            )
            for row in rows
        ]
        if any(tokens != group_tokens[0] for tokens in group_tokens[1:]):
            raise ValueError("rollouts with the same uid have different prompt tokens")
        prompt_key = canonical_prompt_key(tokenizer_fp, template_fp, group_tokens[0])
        cache_row = cache.get(prompt_key)
        if cache_row is None:
            continue
        if [int(token) for token in cache_row["prompt_token_ids"]] != group_tokens[0]:
            raise ValueError("cache prompt tokens do not match the retained prompt key")
        group_id = len(selected_keys)
        selected_rows.extend(rows)
        selected_group_ids.extend([group_id] * rollout_n)
        selected_keys.append(prompt_key)
        floor = float(cache_row["capability_floor"])
        if not 0.0 <= floor <= 1.0:
            raise ValueError("cache capability_floor must be within [0, 1]")
        floors.append(floor)
    if not selected_rows:
        return None
    device = batch.batch.device
    return ProtectedGroupSelection(
        rollout_indices=torch.tensor(selected_rows, dtype=torch.long, device=device),
        group_ids=torch.tensor(selected_group_ids, dtype=torch.long, device=device),
        prompt_keys=tuple(selected_keys),
        capability_floors=torch.tensor(floors, dtype=torch.float32, device=device),
        rollout_count_per_group=rollout_n,
    )


def _validate_right_padding(mask: torch.Tensor) -> None:
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("response_mask must be binary")
    for row in mask:
        zeros = torch.nonzero(row == 0, as_tuple=False)
        if zeros.numel() and bool((row[int(zeros[0].item()) :] != 0).any().item()):
            raise ValueError("response_mask must be right padded")


def build_exact_prefix_batch(
    *,
    batch: DataProto,
    rollout_indices: torch.Tensor,
    reference_budget: int,
    pad_token_id: int,
) -> DataProto:
    """Select exact current-policy token IDs and truncate responses at budget B."""
    if rollout_indices.ndim != 1:
        raise ValueError("rollout_indices must be one-dimensional")
    if (
        rollout_indices.dtype == torch.bool
        or rollout_indices.dtype.is_floating_point
        or rollout_indices.dtype.is_complex
    ):
        raise ValueError("rollout_indices must use an integer dtype")
    if rollout_indices.numel() == 0:
        raise ValueError("rollout_indices must be nonempty")
    indices = rollout_indices.to(dtype=torch.long)
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("rollout_indices must be unique")
    if bool(((indices < 0) | (indices >= len(batch))).any().item()):
        raise ValueError("rollout index is out of range")
    required = ("prompts", "responses", "input_ids", "attention_mask", "response_mask")
    missing = [key for key in required if batch.batch is None or key not in batch.batch]
    if missing:
        raise ValueError(f"batch is missing exact-token fields {missing}")
    prompt_width = batch.batch["prompts"].shape[1]
    response_width = batch.batch["responses"].shape[1]
    if not isinstance(reference_budget, int) or isinstance(reference_budget, bool) or not 0 < reference_budget <= response_width:
        raise ValueError("reference_budget exceeds the response horizon or is invalid")
    expected_input = torch.cat((batch.batch["prompts"], batch.batch["responses"]), dim=-1)
    if not torch.equal(batch.batch["input_ids"], expected_input):
        raise ValueError("input_ids must exactly equal prompt and response token IDs")
    response_mask = batch.batch["response_mask"]
    _validate_right_padding(response_mask)
    prompt_mask_all = batch.batch["attention_mask"][:, :prompt_width]
    for row in range(len(batch)):
        _left_padded_tokens(batch.batch["prompts"][row], prompt_mask_all[row])
    if not torch.equal(batch.batch["attention_mask"][:, prompt_width:], response_mask):
        raise ValueError("attention_mask response segment must equal response_mask")
    if bool((batch.batch["responses"][response_mask == 0] != int(pad_token_id)).any().item()):
        raise ValueError("padded response token IDs must equal pad_token_id")

    selected = indices.to(device=batch.batch["prompts"].device)
    prompts = batch.batch["prompts"][selected].clone()
    responses = batch.batch["responses"][selected, :reference_budget].clone()
    prefix_response_mask = response_mask[selected, :reference_budget].clone()
    prompt_mask = batch.batch["attention_mask"][selected, :prompt_width].clone()
    attention_mask = torch.cat((prompt_mask, prefix_response_mask), dim=-1)
    input_ids = torch.cat((prompts, responses), dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)

    output_keys = set(batch.meta_info.get("reward_extra_keys", ()))
    output_keys.update({"acc", "pred", "score", "reward_score", "reward_extra_info"})
    indices_np = indices.detach().cpu().numpy()
    non_tensors = {
        key: copy.deepcopy(value[indices_np])
        for key, value in batch.non_tensor_batch.items()
        if key not in output_keys
    }
    meta_info = copy.deepcopy(batch.meta_info)
    meta_info.pop("reward_extra_keys", None)
    meta_info["obcf_prefix_scoring"] = True
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": prefix_response_mask,
        },
        non_tensors=non_tensors,
        meta_info=meta_info,
    )
