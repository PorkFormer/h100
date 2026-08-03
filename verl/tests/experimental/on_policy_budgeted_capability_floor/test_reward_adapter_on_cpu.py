from __future__ import annotations

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    NormalizedRewardOutput,
    extract_binary_accuracy,
)
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer


def _scored_batch():
    return DataProto.from_dict(
        tensors={"rm_scores": torch.tensor([[1.0, 0.0], [0.0, -1.0]])},
        non_tensors={"acc": np.asarray([True, False])},
        meta_info={"reward_extra_keys": ["acc"]},
    )


def test_adapter_preserves_current_prescored_reward_shape_without_rescoring():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer._compute_reward_colocate = lambda _batch: pytest.fail("must not rescore")
    batch = _scored_batch()

    output = trainer._score_batch_with_existing_reward_pipeline(batch)

    assert output.reward_tensor is batch.batch["rm_scores"]
    assert output.extra_info["acc"].tolist() == [True, False]


def test_adapter_scores_unscored_prefix_once_even_without_online_rm():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.use_rm = False
    calls = []
    scored = _scored_batch()

    def score(batch):
        calls.append(batch)
        return scored

    trainer._compute_reward_colocate = score
    batch = DataProto.from_dict(
        tensors={"responses": torch.ones((2, 2), dtype=torch.long)},
        meta_info={"obcf_prefix_scoring": True},
    )

    output = trainer._score_batch_with_existing_reward_pipeline(batch)

    assert calls == [batch]
    assert torch.equal(output.reward_tensor, scored.batch["rm_scores"])
    assert output.extra_info["acc"].tolist() == [True, False]


@pytest.mark.parametrize(
    "value",
    [np.asarray([True, False]), [1, 0], torch.tensor([1.0, 0.0])],
)
def test_binary_accuracy_accepts_current_array_shapes(value):
    output = NormalizedRewardOutput(torch.full((2, 3), 99.0), {"acc": value})
    accuracy = extract_binary_accuracy(output, expected_count=2)
    assert accuracy.dtype == torch.float32
    assert accuracy.tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("extra", "count", "message"),
    [
        ({}, 2, "missing"),
        ({"acc": [1]}, 2, "exactly"),
        ({"acc": [[1, 0]]}, 2, "exactly"),
        ({"acc": [1, 0.5]}, 2, "binary"),
        ({"acc": [1, float("nan")]}, 2, "finite"),
        ({"acc": ["yes", "no"]}, 2, "numeric"),
    ],
)
def test_binary_accuracy_fails_closed_and_never_uses_shaped_reward(extra, count, message):
    output = NormalizedRewardOutput(torch.ones((2, 3)), extra)
    with pytest.raises(ValueError, match=message):
        extract_binary_accuracy(output, expected_count=count)
