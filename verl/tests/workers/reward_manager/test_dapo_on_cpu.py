from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_manager.dapo import DAPORewardManager


class _Tokenizer:
    eos_token = "<eos>"

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return "prompt" if len(token_ids) == 1 else "response"


def test_exact_score_plus_overlong_reward_equals_terminal_total_reward():
    response_length = 2048
    prompts = torch.tensor([[1]], dtype=torch.long)
    responses = torch.full((1, response_length), 2, dtype=torch.long)
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "attention_mask": torch.ones((1, 1 + response_length), dtype=torch.long),
            },
            batch_size=1,
        ),
        non_tensor_batch={
            "data_source": np.asarray(["math_dapo"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "42"}], dtype=object),
        },
    )

    def score_fn(**kwargs):
        assert kwargs["data_source"] == "math_dapo"
        return {"score": -1.0, "acc": 0.0}

    manager = DAPORewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        compute_score=score_fn,
        max_resp_len=response_length,
        overlong_buffer_cfg=SimpleNamespace(enable=True, len=410, penalty_factor=1.0, log=True),
    )
    result = manager(data, return_dict=True)

    score = result["reward_extra_info"]["score"]
    overlong_reward = result["reward_extra_info"]["overlong_reward"]
    assert score == [-1.0]
    assert overlong_reward == pytest.approx([-1.0])
    reward = result["reward_tensor"]
    assert reward[0, -1].item() == pytest.approx(-2.0)
    assert reward[0, -1].item() == pytest.approx(score[0] + overlong_reward[0])
    assert torch.count_nonzero(reward[0, :-1]).item() == 0
