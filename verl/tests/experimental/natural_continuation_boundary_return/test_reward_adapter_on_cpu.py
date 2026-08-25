from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BoundaryRewardOutput,
    apply_boundary_return,
    build_long_reward_batch,
    extract_required_reward_scalars,
)
from verl.experimental.natural_continuation_boundary_return.runtime import (
    BoundaryContinuationCapture,
    BoundaryContinuationGeneration,
)
from verl.workers.config.rollout import BoundaryReturnConfig


def _candidate() -> DataProto:
    prompts = torch.tensor(
        [[0, 11, 12], [0, 21, 22], [0, 31, 32], [0, 41, 42], [0, 51, 52]],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [
            [101, 102, 103, 104],
            [111, 112, 113, 114],
            [121, 122, 123, 124],
            [131, 132, 133, 134],
            [141, 142, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    shaped = torch.tensor(
        [
            [-0.1, 0.0, 0.0, 0.1],
            [-0.1, 0.0, 0.0, 0.1],
            [-0.1, 0.0, 0.0, 1.1],
            [-0.1, 0.0, 0.0, 1.1],
            [0.0, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention_mask,
            "position_ids": torch.clamp(attention_mask.cumsum(-1) - 1, min=0),
            "response_mask": attention_mask[:, 3:].clone(),
            "token_level_scores": shaped.clone(),
            "token_level_rewards": shaped.clone(),
        },
        non_tensors={
            "uid": np.asarray(["a", "b", "c", "d", "e"], dtype=object),
            "trajectory_id": np.asarray(["a:0", "b:0", "c:0", "d:0", "e:0"], dtype=object),
            "data_source": np.asarray(["math"] * 5, dtype=object),
            "reward_model": np.asarray(
                [{"ground_truth": str(index)} for index in range(5)], dtype=object
            ),
            "extra_info": np.asarray([{"index": index} for index in range(5)], dtype=object),
            "difficulty": np.asarray(["hard", "easy", "hard", "easy", "medium"], dtype=object),
            "acc": np.asarray([0.0, 0.0, 1.0, 1.0, 1.0]),
            "score": np.asarray([0.0, 0.0, 1.0, 1.0, 0.2]),
            "overlong_reward": np.asarray([-0.1, -0.1, -0.1, -0.1, 0.0]),
            "boundary_stale": np.asarray([99] * 5),
        },
        meta_info={
            "reward_extra_keys": ["acc", "score", "overlong_reward"],
            "verifier_protocol": "fake-v1",
            "boundary_internal": "remove",
        },
    )


def _capture() -> BoundaryContinuationCapture:
    generations = []
    for parent, tail in zip(range(4), ((201,), (211, 212), (), (231,)), strict=True):
        generations.append(
            BoundaryContinuationGeneration(
                parent_index=parent,
                request_id=f"r{parent}",
                branch_id=0,
                uid=chr(ord("a") + parent),
                trajectory_id=f"{chr(ord('a') + parent)}:0",
                prompt_token_ids=(11 + parent * 10, 12 + parent * 10),
                prefix_token_ids=tuple(range(101 + parent * 10, 105 + parent * 10)),
                tail_token_ids=tail,
                actual_policy_version=7,
            )
        )
    return BoundaryContinuationCapture(
        hit_response_cap=np.asarray([True, True, True, True, False]),
        requests=(),
        generations=tuple(generations),
        normal_response_tokens=18,
    )


def _long_output(*, shaped_value=99999.0) -> BoundaryRewardOutput:
    # H wrong -> L wrong, H wrong -> L correct, H correct -> L correct,
    # H correct -> L wrong. Task scores intentionally include a negative delta.
    return BoundaryRewardOutput(
        reward_tensor=torch.full((4, 6), shaped_value, dtype=torch.float32),
        extra_info={
            "acc": np.asarray([0.0, 1.0, 1.0, 0.0]),
            "score": np.asarray([0.0, 1.0, 1.0, -1.0]),
        },
    )


def _config(mode="replace") -> BoundaryReturnConfig:
    return BoundaryReturnConfig(mode=mode, long_response_length=6)


def test_long_reward_batch_uses_original_prompt_and_full_response_and_copies_verifier_metadata():
    batch = build_long_reward_batch(_candidate(), _capture().generations, pad_token_id=0)

    assert batch.batch["prompts"][0].tolist() == [11, 12]
    assert batch.batch["responses"][0].tolist() == [101, 102, 103, 104, 201, 0]
    assert batch.batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    assert batch.non_tensor_batch["difficulty"].tolist() == ["hard", "easy", "hard", "easy"]
    assert batch.non_tensor_batch["extra_info"][2] == {"index": 2}
    assert batch.non_tensor_batch["boundary_parent_index"].tolist() == [0, 1, 2, 3]
    assert batch.meta_info == {"verifier_protocol": "fake-v1"}
    for excluded in ("acc", "score", "overlong_reward", "boundary_stale"):
        assert excluded not in batch.non_tensor_batch


def test_fake_verifier_requires_extra_custom_metadata_proving_full_copy():
    batch = build_long_reward_batch(_candidate(), _capture().generations, pad_token_id=0)

    def fake_verifier(data):
        assert data.non_tensor_batch["difficulty"].tolist() == ["hard", "easy", "hard", "easy"]
        return _long_output()

    assert fake_verifier(batch).extra_info["acc"].tolist() == [0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (BoundaryRewardOutput(torch.ones((2, 2)), {}), "missing"),
        (BoundaryRewardOutput(torch.ones((2, 2)), {"acc": [0, 1], "score": [1]}), "exactly"),
        (
            BoundaryRewardOutput(torch.ones((2, 2)), {"acc": [0, float("nan")], "score": [0, 1]}),
            "finite",
        ),
        (BoundaryRewardOutput(torch.ones((2, 2)), {"acc": [0, 1], "score": [0, "bad"]}), "scalar"),
    ],
)
def test_required_reward_scalars_fail_closed_including_scalar_reward_manager(output, message):
    with pytest.raises(ValueError, match=message):
        extract_required_reward_scalars(
            output,
            expected_count=2,
            correctness_key="acc",
            task_score_key="score",
        )


def test_shadow_keeps_candidate_bitwise_unchanged_and_returns_isolated_arrays():
    candidate = _candidate()
    tensor_before = {key: value.clone() for key, value in candidate.batch.items()}
    non_tensor_before = {key: value.copy() for key, value in candidate.non_tensor_batch.items()}
    keys_before = set(candidate.batch.keys()), set(candidate.non_tensor_batch)

    result = apply_boundary_return(
        candidate,
        capture=_capture(),
        long_reward_output=_long_output(),
        config=_config("shadow"),
    )

    assert (set(candidate.batch.keys()), set(candidate.non_tensor_batch)) == keys_before
    for key, value in tensor_before.items():
        assert torch.equal(candidate.batch[key], value)
    for key, value in non_tensor_before.items():
        assert candidate.non_tensor_batch[key].tolist() == value.tolist()
    assert result.boundary_acc.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert result.boundary_task_score.tolist() == [0.0, 1.0, 1.0, -1.0, 0.2]
    assert "boundary_acc" not in candidate.non_tensor_batch


def test_replace_preserves_raw_fields_and_prefix_shaping_residual_and_syncs_rewards():
    candidate = _candidate()
    original_scores = candidate.batch["token_level_scores"].clone()
    original_acc = candidate.non_tensor_batch["acc"].copy()
    original_task = candidate.non_tensor_batch["score"].copy()

    result = apply_boundary_return(
        candidate,
        capture=_capture(),
        long_reward_output=_long_output(shaped_value=1.0e30),
        config=_config("replace"),
    )

    assert candidate.non_tensor_batch["acc"].tolist() == original_acc.tolist()
    assert candidate.non_tensor_batch["score"].tolist() == original_task.tolist()
    assert candidate.non_tensor_batch["boundary_acc"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert candidate.non_tensor_batch["boundary_task_score"].tolist() == [0.0, 1.0, 1.0, -1.0, 0.2]
    # Only row 1 (+1) and row 3 (-2) change; huge long shaped rewards are ignored.
    expected = original_scores.clone()
    expected[1, 3] += 1.0
    expected[3, 3] -= 2.0
    assert torch.equal(candidate.batch["token_level_scores"], expected)
    assert torch.equal(candidate.batch["token_level_rewards"], expected)
    assert candidate.batch["token_level_scores"].data_ptr() != candidate.batch["token_level_rewards"].data_ptr()
    assert result.metrics["boundary_return/prefix_penalty_drift_max"] == 0.0
    assert result.metrics["boundary_return/regressed_count"] == 1.0
    assert result.metrics["boundary_return/regressed_rate_given_cap_success"] == 0.5
    assert result.metrics["boundary_return/recovered_count"] == 1.0
    assert result.metrics["boundary_return/recovered_rate_given_cap_failure"] == 0.5
    assert result.metrics["boundary_return/transition_h_wrong_l_wrong_count"] == 1.0
    assert result.metrics["boundary_return/transition_h_wrong_l_correct_count"] == 1.0
    assert result.metrics["boundary_return/transition_h_correct_l_correct_count"] == 1.0
    assert result.metrics["boundary_return/transition_h_correct_l_wrong_count"] == 1.0
    assert result.task_score_delta[3] == -2.0


def test_replacement_rejects_missing_cap_score_duplicate_parent_and_empty_prefix_mask():
    capture = _capture()
    with pytest.raises(ValueError, match="exactly one long score"):
        apply_boundary_return(
            _candidate(),
            capture=SimpleNamespace(
                hit_response_cap=capture.hit_response_cap,
                generations=capture.generations[:-1],
                normal_response_tokens=18,
            ),
            long_reward_output=BoundaryRewardOutput(
                reward_tensor=torch.ones((3, 6)), extra_info={"acc": [0, 1, 1], "score": [0, 1, 1]}
            ),
            config=_config(),
        )

    duplicate = list(capture.generations)
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate long score"):
        apply_boundary_return(
            _candidate(),
            capture=SimpleNamespace(
                hit_response_cap=capture.hit_response_cap,
                generations=tuple(duplicate),
                normal_response_tokens=18,
            ),
            long_reward_output=_long_output(),
            config=_config(),
        )

    candidate = _candidate()
    candidate.batch["response_mask"][0].zero_()
    with pytest.raises(ValueError, match="empty response prefix"):
        apply_boundary_return(
            candidate,
            capture=capture,
            long_reward_output=_long_output(),
            config=_config(),
        )
