import math

import pytest
import torch

from verl import DataProto
from verl.experimental.success_support_floor.batch import (
    build_augmented_actor_batch,
    build_support_batch,
)


def _rl_batch():
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 10, 11], [0, 12, 13]]),
            "responses": torch.tensor([[20, 2, 0, 0], [21, 22, 2, 0]]),
            "input_ids": torch.tensor([[0, 10, 11, 20, 2, 0, 0], [0, 12, 13, 21, 22, 2, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 3, 0, 0], [0, 0, 1, 2, 3, 4, 0]]),
            "response_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            "old_log_probs": torch.full((2, 4), -1.0),
            "advantages": torch.ones(2, 4),
        }
    )


def test_support_padding_masks_and_position_ids():
    support = build_support_batch(
        prompt_tokens_by_key={"a": [7, 8], "b": [9]},
        witnesses=[
            {"prompt_key": "a", "response_token_ids": [30, 2], "reference_seq_logprob": -3.0},
            {"prompt_key": "b", "response_token_ids": [31, 32, 2], "reference_seq_logprob": -4.0},
        ],
        prompt_width=3,
        response_width=4,
        pad_token_id=0,
    )

    assert support.batch["prompts"].tolist() == [[0, 7, 8], [0, 0, 9]]
    assert support.batch["responses"].tolist() == [[30, 2, 0, 0], [31, 32, 2, 0]]
    assert support.batch["response_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]
    assert support.batch["position_ids"].tolist() == [
        [0, 0, 1, 2, 3, 3, 3],
        [0, 0, 0, 1, 2, 3, 3],
    ]


def test_augmented_batch_masks_are_disjoint_and_rl_batch_is_unchanged():
    rl = _rl_batch()
    before = {key: value.clone() for key, value in rl.batch.items()}
    support = build_support_batch(
        prompt_tokens_by_key={"a": [7, 8]},
        witnesses=[{"prompt_key": "a", "response_token_ids": [30, 2], "reference_seq_logprob": -3.0}],
        prompt_width=3,
        response_width=4,
        pad_token_id=0,
    )
    augmented = build_augmented_actor_batch(
        rl,
        support,
        lambda_value=0.25,
        alpha=0.5,
        global_support_batch_size=1,
    )

    assert len(augmented) == 3
    assert augmented.batch["ppo_response_mask"].sum().item() == 5
    assert augmented.batch["support_response_mask"].sum().item() == 2
    assert not bool(
        (augmented.batch["ppo_response_mask"].bool() & augmented.batch["support_response_mask"].bool())
        .any()
        .item()
    )
    assert augmented.batch["support_sample_mask"].tolist() == [False, False, True]
    assert augmented.batch["support_lambda"].tolist() == [0.25, 0.25, 0.25]
    assert augmented.batch["support_log_alpha"].tolist() == pytest.approx([math.log(0.5)] * 3)
    for key, value in before.items():
        assert torch.equal(rl.batch[key], value)
