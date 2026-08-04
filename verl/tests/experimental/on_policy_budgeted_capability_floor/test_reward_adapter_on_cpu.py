from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    NormalizedRewardOutput,
    extract_binary_accuracy,
    verifier_pipeline_fingerprint,
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


@pytest.mark.parametrize(
    "extra",
    [
        {"acc": [0, 1], "error": ["verifier crashed", None]},
        {"acc": [0, 1], "timeout": [True, False]},
    ],
)
def test_binary_accuracy_rejects_rowwise_verifier_failures(extra):
    output = NormalizedRewardOutput(torch.ones((2, 3)), extra)
    with pytest.raises(ValueError, match="error|timeout"):
        extract_binary_accuracy(output, expected_count=2)


def test_binary_accuracy_accepts_explicit_numpy_false_timeout_flags():
    output = NormalizedRewardOutput(
        torch.ones((2, 3)),
        {"acc": [0, 1], "timeout": np.asarray([False, False])},
    )
    assert extract_binary_accuracy(output, expected_count=2).tolist() == [0.0, 1.0]


def test_verifier_pipeline_fingerprint_is_deterministic_and_config_sensitive():
    first = verifier_pipeline_fingerprint(reward_manager_name="naive")
    assert len(first) == 64
    assert first == verifier_pipeline_fingerprint(reward_manager_name="naive")
    assert first != verifier_pipeline_fingerprint(reward_manager_name="dapo")
    assert first != verifier_pipeline_fingerprint(
        reward_manager_name="naive", reward_manager_source="importlib",
        reward_manager_module_path=__file__, reward_manager_module_name="Manager",
    )
    importlib_first = verifier_pipeline_fingerprint(
        reward_manager_name="naive", reward_manager_source="importlib",
        reward_manager_module_path=__file__, reward_manager_module_name="Manager",
    )
    assert importlib_first != verifier_pipeline_fingerprint(
        reward_manager_name="naive", reward_manager_source="importlib",
        reward_manager_module_path=__file__, reward_manager_module_name="DifferentManager",
    )
    assert first != verifier_pipeline_fingerprint(
        reward_manager_name="naive", custom_reward_kwargs={"pass_rate": 0.5}
    )
    assert first != verifier_pipeline_fingerprint(
        reward_manager_name="naive", reward_kwargs={"num_examine": 1}
    )
    assert first != verifier_pipeline_fingerprint(
        reward_manager_name="naive",
        sandbox_fusion={"url": "https://verifier.invalid", "memory_limit_mb": 512},
    )


def test_verifier_pipeline_fingerprint_normalizes_hydra_list_config():
    hydra = OmegaConf.create({"stops": ["a", "b"]})
    assert verifier_pipeline_fingerprint(custom_reward_kwargs=hydra) == (
        verifier_pipeline_fingerprint(custom_reward_kwargs={"stops": ["a", "b"]})
    )
