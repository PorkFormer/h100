from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.capability_constraints.identity import canonical_prompt_key
from verl.experimental.on_policy_budgeted_capability_floor.prefix_batch import (
    build_exact_prefix_batch,
    resolve_protected_groups,
)


TOKENIZER_FINGERPRINT = "tokenizer"
TEMPLATE_FINGERPRINT = "template"


class _Cache:
    def __init__(self, rows):
        self.manifest = {
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "chat_template_fingerprint": TEMPLATE_FINGERPRINT,
        }
        self._rows = {row["prompt_key"]: dict(row) for row in rows}

    def get(self, prompt_key):
        row = self._rows.get(prompt_key)
        return None if row is None else dict(row)


def _cache_row(tokens, floor):
    return {
        "prompt_key": canonical_prompt_key(
            TOKENIZER_FINGERPRINT, TEMPLATE_FINGERPRINT, tokens
        ),
        "prompt_token_ids": list(tokens),
        "capability_floor": floor,
    }


def _group_batch(*, uids=("first", "first", "second", "second")):
    prompts = torch.tensor(
        [
            [0, 0, 10, 11],
            [0, 0, 10, 11],
            [0, 12, 13, 14],
            [0, 12, 13, 14],
        ]
    )
    prompt_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    response_mask = torch.ones(4, 3, dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": torch.tensor([[20, 21, 22]] * 4),
            "attention_mask": torch.cat((prompt_mask, response_mask), dim=-1),
        },
        non_tensors={"uid": np.asarray(uids, dtype=object)},
    )


def test_resolve_uses_exact_left_padded_prompt_tokens_and_skips_unprotected_groups():
    batch = _group_batch()
    cache = _Cache([_cache_row([10, 11], 0.125)])

    selection = resolve_protected_groups(batch=batch, cache=cache, rollout_n=2)

    assert selection is not None
    assert selection.rollout_indices.tolist() == [0, 1]
    assert selection.group_ids.tolist() == [0, 0]
    assert selection.prompt_keys == (
        canonical_prompt_key(TOKENIZER_FINGERPRINT, TEMPLATE_FINGERPRINT, [10, 11]),
    )
    assert selection.capability_floors.tolist() == pytest.approx([0.125])
    assert selection.rollout_count_per_group == 2


def test_duplicate_prompt_occurrences_with_distinct_uids_remain_separate_groups():
    batch = _group_batch(uids=("occurrence-a", "occurrence-a", "occurrence-b", "occurrence-b"))
    batch.batch["prompts"][2:] = batch.batch["prompts"][:2]
    batch.batch["attention_mask"][2:, :4] = batch.batch["attention_mask"][:2, :4]
    cache = _Cache([_cache_row([10, 11], 0.5)])

    selection = resolve_protected_groups(batch=batch, cache=cache, rollout_n=2)

    assert selection is not None
    assert selection.rollout_indices.tolist() == [0, 1, 2, 3]
    assert selection.group_ids.tolist() == [0, 0, 1, 1]
    assert selection.prompt_keys[0] == selection.prompt_keys[1]
    assert selection.capability_floors.tolist() == pytest.approx([0.5, 0.5])


def test_resolve_order_is_first_uid_occurrence_even_when_rows_are_interleaved():
    batch = _group_batch(uids=("b", "a", "b", "a"))
    batch.batch["prompts"][[0, 2]] = batch.batch["prompts"][[2, 3]]
    batch.batch["attention_mask"][[0, 2], :4] = batch.batch["attention_mask"][[2, 3], :4]
    batch.batch["prompts"][3] = batch.batch["prompts"][1]
    batch.batch["attention_mask"][3, :4] = batch.batch["attention_mask"][1, :4]
    cache = _Cache([_cache_row([12, 13, 14], 0.25), _cache_row([10, 11], 0.75)])

    selection = resolve_protected_groups(batch=batch, cache=cache, rollout_n=2)

    assert selection is not None
    assert selection.rollout_indices.tolist() == [0, 2, 1, 3]
    assert selection.group_ids.tolist() == [0, 0, 1, 1]
    assert selection.capability_floors.tolist() == pytest.approx([0.25, 0.75])


def test_resolve_returns_none_when_no_prompt_is_protected():
    assert resolve_protected_groups(batch=_group_batch(), cache=_Cache([]), rollout_n=2) is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda batch: batch.non_tensor_batch.pop("uid"), "uid"),
        (
            lambda batch: batch.non_tensor_batch.__setitem__(
                "uid", np.asarray(["a", "a", "a", "b"], dtype=object)
            ),
            "exactly rollout_n",
        ),
        (
            lambda batch: batch.batch["prompts"].__setitem__((1, -1), 99),
            "different prompt tokens",
        ),
        (
            lambda batch: batch.batch["attention_mask"].__setitem__((0, slice(0, 4)), torch.tensor([0, 1, 0, 1])),
            "left padded",
        ),
    ],
)
def test_resolve_rejects_malformed_group_identity(mutate, message):
    batch = _group_batch()
    mutate(batch)
    cache = _Cache([_cache_row([10, 11], 0.125)])
    with pytest.raises(ValueError, match=message):
        resolve_protected_groups(batch=batch, cache=cache, rollout_n=2)


def test_resolve_rejects_cache_row_whose_tokens_do_not_match_its_key():
    row = _cache_row([10, 11], 0.125)
    row["prompt_token_ids"] = [99]
    with pytest.raises(ValueError, match="cache prompt tokens"):
        resolve_protected_groups(batch=_group_batch(), cache=_Cache([row]), rollout_n=2)


def _prefix_source_batch():
    prompts = torch.tensor([[0, 10, 11], [0, 12, 13], [0, 14, 15]])
    responses = torch.tensor(
        [
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 0, 0, 0],
            [40, 41, 42, 43, 44, 45],
        ]
    )
    prompt_mask = torch.tensor([[0, 1, 1]] * 3)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    input_ids = torch.cat((prompts, responses), dim=-1)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
            "response_mask": response_mask,
            "rm_scores": torch.arange(18, dtype=torch.float32).reshape(3, 6),
            "token_level_scores": torch.ones(3, 6),
            "advantages": torch.ones(3, 6),
        },
        non_tensors={
            "uid": np.asarray(["a", "b", "c"], dtype=object),
            "data_source": np.asarray(["math"] * 3, dtype=object),
            "reward_model": np.asarray([{"ground_truth": str(i)} for i in range(3)], dtype=object),
            "extra_info": np.asarray([{"row": i} for i in range(3)], dtype=object),
            "verifier_context": np.asarray([{"keep": i} for i in range(3)], dtype=object),
            "acc": np.asarray([1.0, 0.0, 1.0]),
            "pred": np.asarray(["x", "y", "z"], dtype=object),
        },
        meta_info={
            "reward_extra_keys": ["acc", "pred"],
            "temperature": 1.0,
        },
    )


def test_exact_prefix_batch_selects_and_truncates_without_token_transformation():
    source = _prefix_source_batch()

    prefix = build_exact_prefix_batch(
        batch=source,
        rollout_indices=torch.tensor([2, 0]),
        reference_budget=4,
        pad_token_id=0,
    )

    assert prefix.batch["prompts"].tolist() == [[0, 14, 15], [0, 10, 11]]
    assert prefix.batch["responses"].tolist() == [[40, 41, 42, 43], [20, 21, 22, 23]]
    assert prefix.batch["input_ids"].tolist() == [
        [0, 14, 15, 40, 41, 42, 43],
        [0, 10, 11, 20, 21, 22, 23],
    ]
    assert prefix.batch["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 1, 1]] * 2
    assert prefix.batch["response_mask"].tolist() == [[1, 1, 1, 1]] * 2
    assert prefix.batch["position_ids"].tolist() == [[0, 0, 1, 2, 3, 4, 5]] * 2
    assert set(prefix.batch.keys()) == {
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "position_ids",
        "response_mask",
    }


def test_exact_prefix_preserves_padding_and_deep_copied_verifier_inputs_but_strips_reward_outputs():
    source = _prefix_source_batch()
    before_extra = copy.deepcopy(source.non_tensor_batch["extra_info"].tolist())

    prefix = build_exact_prefix_batch(
        batch=source,
        rollout_indices=torch.tensor([1]),
        reference_budget=5,
        pad_token_id=0,
    )

    assert prefix.batch["responses"].tolist() == [[30, 31, 32, 0, 0]]
    assert prefix.batch["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 1, 0, 0]]
    assert prefix.batch["response_mask"].tolist() == [[1, 1, 1, 0, 0]]
    assert prefix.batch["position_ids"].tolist() == [[0, 0, 1, 2, 3, 4, 4, 4]]
    assert prefix.non_tensor_batch["reward_model"].tolist() == [{"ground_truth": "1"}]
    assert prefix.non_tensor_batch["extra_info"].tolist() == [{"row": 1}]
    assert prefix.non_tensor_batch["verifier_context"].tolist() == [{"keep": 1}]
    assert "acc" not in prefix.non_tensor_batch
    assert "pred" not in prefix.non_tensor_batch
    assert "reward_extra_keys" not in prefix.meta_info
    assert prefix.meta_info["temperature"] == 1.0

    prefix.non_tensor_batch["extra_info"][0]["mutated"] = True
    assert source.non_tensor_batch["extra_info"].tolist() == before_extra


def test_exact_prefix_does_not_mutate_any_source_tensor_or_metadata():
    source = _prefix_source_batch()
    tensor_before = {key: value.clone() for key, value in source.batch.items()}
    non_tensor_before = copy.deepcopy(source.non_tensor_batch)
    meta_before = copy.deepcopy(source.meta_info)

    build_exact_prefix_batch(
        batch=source,
        rollout_indices=torch.tensor([0, 2]),
        reference_budget=2,
        pad_token_id=0,
    )

    for key, value in tensor_before.items():
        assert torch.equal(source.batch[key], value)
    for key, value in non_tensor_before.items():
        assert source.non_tensor_batch[key].tolist() == value.tolist()
    assert source.meta_info == meta_before


@pytest.mark.parametrize(
    ("indices", "budget", "message"),
    [
        (torch.tensor([[0]]), 2, "one-dimensional"),
        (torch.tensor([0, 0]), 2, "unique"),
        (torch.tensor([3]), 2, "out of range"),
        (torch.tensor([0]), 7, "response horizon"),
        (torch.tensor([], dtype=torch.long), 2, "nonempty"),
    ],
)
def test_exact_prefix_rejects_invalid_selection_or_budget(indices, budget, message):
    with pytest.raises(ValueError, match=message):
        build_exact_prefix_batch(
            batch=_prefix_source_batch(),
            rollout_indices=indices,
            reference_budget=budget,
            pad_token_id=0,
        )


def test_exact_prefix_rejects_inconsistent_text_masks():
    source = _prefix_source_batch()
    source.batch["response_mask"][0, 1] = 0
    with pytest.raises(ValueError, match="response_mask"):
        build_exact_prefix_batch(
            batch=source,
            rollout_indices=torch.tensor([0]),
            reference_budget=4,
            pad_token_id=0,
        )
