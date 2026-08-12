from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.forced_answer_probe import (
    ForcedAnswerGeneration,
    ForcedAnswerProbeCapture,
    apply_terminal_reward_targets,
    build_fa_tr_training_credit_result,
    build_probe_reward_batch,
    compute_fa_tr_credit_targets,
    compute_pfa_by_parent,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.config import ForcedAnswerProbeConfig, ForcedAnswerTrainingCreditConfig


@pytest.mark.parametrize(
    ("success_count", "expected_target"),
    [(0, None), (1, None), (2, None), (3, 0.5), (4, 1.0)],
)
def test_k4_threshold_mapping(success_count, expected_target):
    pfa = compute_pfa_by_parent({0: [True] * success_count + [False] * (4 - success_count)})
    result = compute_fa_tr_credit_targets(
        hit_response_cap=[True],
        probe_attempted=[True],
        context_overflow=[False],
        original_correctness=[0.0],
        pfa_by_parent=pfa,
        correctness_threshold=0.5,
        activation_threshold=0.75,
    )
    if expected_target is None:
        assert result.target_reward_by_parent == {}
    else:
        assert result.target_reward_by_parent == {0: expected_target}


@pytest.mark.parametrize(
    ("hit_cap", "attempted", "overflow", "original_correct"),
    [(True, True, False, 1.0), (False, True, False, 0.0), (True, False, True, 0.0)],
)
def test_ineligible_trajectory_is_never_corrected(hit_cap, attempted, overflow, original_correct):
    result = compute_fa_tr_credit_targets(
        hit_response_cap=[hit_cap],
        probe_attempted=[attempted],
        context_overflow=[overflow],
        original_correctness=[original_correct],
        pfa_by_parent={0: 1.0} if attempted else {},
        correctness_threshold=0.5,
        activation_threshold=0.75,
    )
    assert result.eligible_parent_indices == ()
    assert result.target_reward_by_parent == {}


def test_terminal_delta_replaces_scalar_and_preserves_other_tokens():
    original = torch.tensor([[0.25, -0.5, 0.0, -2.0]])
    effective = apply_terminal_reward_targets(
        original_reward_tensor=original,
        response_mask=torch.ones_like(original),
        current_row_to_parent=[0],
        target_reward_by_parent={0: 0.5},
    )
    assert torch.equal(effective[:, :-1], original[:, :-1])
    assert effective.sum().item() == pytest.approx(0.5)
    assert effective[0, -1].item() == pytest.approx(0.75)
    assert torch.equal(original, torch.tensor([[0.25, -0.5, 0.0, -2.0]]))


def test_simple_terminal_replacement_and_padding():
    original = torch.tensor([[0.0, 0.0, -2.0, 0.0, 0.0]])
    effective = apply_terminal_reward_targets(
        original_reward_tensor=original,
        response_mask=torch.tensor([[1, 1, 1, 0, 0]]),
        current_row_to_parent=[7],
        target_reward_by_parent={7: 0.5},
    )
    assert torch.equal(effective, torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0]]))


def test_balance_permutation_maps_parent_target_to_current_row():
    original = torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, -2.0], [0.0, 0.0, -2.0]])
    effective = apply_terminal_reward_targets(
        original_reward_tensor=original,
        response_mask=torch.ones_like(original),
        current_row_to_parent=[2, 0, 1],
        target_reward_by_parent={1: 0.5},
    )
    assert torch.equal(effective[:2], original[:2])
    assert effective[2].sum().item() == pytest.approx(0.5)


def test_target_for_filtered_out_parent_fails_closed():
    with pytest.raises(ValueError, match="do not belong to the current PPO batch"):
        apply_terminal_reward_targets(
            original_reward_tensor=torch.zeros((2, 2)),
            response_mask=torch.ones((2, 2)),
            current_row_to_parent=[2, 0],
            target_reward_by_parent={1: 0.5},
        )


def test_disabled_path_is_bitwise_vanilla_and_reports_no_corrections():
    original = torch.tensor([[0.0, 0.0, -2.0]])
    result = build_fa_tr_training_credit_result(
        original_reward_tensor=original,
        response_mask=torch.ones_like(original),
        current_row_to_parent=[0],
        current_uids=["group-a"],
        hit_response_cap=[True],
        probe_attempted=[True],
        context_overflow=[False],
        original_correctness=[0.0],
        successes_by_parent={0: [True, True, True, True]},
        correctness_threshold=0.5,
        enable=False,
        activation_threshold=0.75,
    )
    assert result.effective_reward_tensor is original
    assert torch.equal(result.effective_reward_tensor, original)
    assert result.metrics["fa_tr/num_reward_corrected"] == 0.0
    assert result.metrics["fa_tr/reward_correction_rate"] == 0.0


def test_group_metrics_count_only_groups_with_a_correction():
    original = torch.full((8, 2), -1.0)
    result = build_fa_tr_training_credit_result(
        original_reward_tensor=original,
        response_mask=torch.ones_like(original),
        current_row_to_parent=list(range(8)),
        current_uids=["a"] * 4 + ["b"] * 4,
        hit_response_cap=[True] + [False] * 7,
        probe_attempted=[True] + [False] * 7,
        context_overflow=[False] * 8,
        original_correctness=[0.0] * 8,
        successes_by_parent={0: [True] * 4},
        correctness_threshold=0.5,
        enable=True,
        activation_threshold=0.75,
    )
    assert result.metrics["fa_tr/num_groups"] == 2.0
    assert result.metrics["fa_tr/num_groups_with_correction"] == 1.0
    assert result.metrics["fa_tr/group_correction_rate"] == 0.5
    assert result.metrics["fa_tr/original_reward_mean_corrected_subset"] == -2.0
    assert result.metrics["fa_tr/effective_reward_mean_corrected_subset"] == 1.0
    assert result.metrics["fa_tr/reward_delta_mean"] == 3.0
    assert result.metrics["fa_tr/reward_delta_max"] == 3.0


def test_pfa_distribution_metrics_use_eligible_subset():
    result = build_fa_tr_training_credit_result(
        original_reward_tensor=torch.zeros((5, 1)),
        response_mask=torch.ones((5, 1)),
        current_row_to_parent=list(range(5)),
        current_uids=[f"group-{index}" for index in range(5)],
        hit_response_cap=[True] * 5,
        probe_attempted=[True] * 5,
        context_overflow=[False] * 5,
        original_correctness=[0.0] * 5,
        successes_by_parent={
            parent: [True] * parent + [False] * (4 - parent) for parent in range(5)
        },
        correctness_threshold=0.5,
        enable=True,
        activation_threshold=0.75,
    )
    metrics = result.metrics
    assert metrics["fa_tr/pfa_mean"] == 0.5
    assert metrics["fa_tr/pfa_eq_0_rate"] == 0.2
    assert metrics["fa_tr/pfa_ge_025_rate"] == 0.8
    assert metrics["fa_tr/pfa_ge_050_rate"] == 0.6
    assert metrics["fa_tr/pfa_ge_075_rate"] == 0.4
    assert metrics["fa_tr/pfa_eq_1_rate"] == 0.2
    assert metrics["fa_tr/num_reward_corrected"] == 2.0
    assert metrics["fa_tr/reward_correction_rate_given_eligible"] == 0.4


def test_training_credit_config_defaults_and_fail_fast_validation():
    config = ForcedAnswerProbeConfig()
    assert config.num_samples == 4
    assert config.training_credit.enable is False
    assert config.training_credit.activation_threshold == 0.75
    assert config.training_credit.reward_mode == "centered_pfa"
    config.validate()

    with pytest.raises(ValueError, match="activation_threshold"):
        ForcedAnswerTrainingCreditConfig(activation_threshold=1.01).validate()
    with pytest.raises(ValueError, match="reward_mode"):
        ForcedAnswerTrainingCreditConfig(reward_mode="additive").validate()
    with pytest.raises(ValueError, match="requires forced_answer_probe.enable=true"):
        ForcedAnswerProbeConfig(
            enable=False,
            training_credit=ForcedAnswerTrainingCreditConfig(enable=True),
        ).validate()


def test_trainer_probe_enable_check_fails_fast_for_invalid_credit_dependency():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                forced_answer_probe=ForcedAnswerProbeConfig(
                    enable=False,
                    training_credit=ForcedAnswerTrainingCreditConfig(enable=True),
                )
            )
        )
    )
    with pytest.raises(ValueError, match="requires forced_answer_probe.enable=true"):
        trainer._forced_answer_probe_enabled()


def _integration_batch() -> DataProto:
    responses = torch.tensor([[11, 12, 13, 0], [21, 22, 23, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    prompts = torch.tensor([[1], [2]], dtype=torch.long)
    attention_mask = torch.cat((torch.ones_like(prompts), response_mask), dim=-1)
    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=-1),
                "attention_mask": attention_mask,
                "position_ids": torch.clamp(attention_mask.cumsum(-1) - 1, min=0),
                "response_mask": response_mask,
                "rm_scores": torch.tensor([[0.0, 0.0, -2.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
                "old_log_probs": torch.zeros((2, 4)),
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "uid": np.asarray(["group-a", "group-b"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "a"}, {"ground_truth": "b"}], dtype=object),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "__forced_answer_probe_parent_index__": np.asarray([1, 0]),
        },
    )


def test_score_integration_corrects_balanced_row_without_mutating_actor_or_rm_tensors():
    batch = _integration_batch()
    actor_keys = ("responses", "input_ids", "attention_mask", "position_ids", "response_mask", "old_log_probs")
    snapshot = {key: batch.batch[key].clone() for key in (*actor_keys, "rm_scores")}
    generations = tuple(
        ForcedAnswerGeneration(0, branch, (1,), (40 + branch,)) for branch in range(4)
    )
    capture = ForcedAnswerProbeCapture(
        hit_response_cap=np.asarray([True, False]),
        probe_attempted=np.asarray([True, False]),
        context_overflow=np.asarray([False, False]),
        generations=generations,
        probe_input_tokens=1,
    )
    probe_reward_batch = build_probe_reward_batch(batch, generations, pad_token_id=0)

    class _RewardLoopManager:
        reward_loop_workers = [object()]

        def compute_rm_score(self, data):
            branch_ids = np.asarray(data.non_tensor_batch["probe_branch_id"])
            acc = (branch_ids < 3).astype(np.float32)
            return DataProto(
                batch=TensorDict(
                    {"rm_scores": torch.zeros_like(data.batch["responses"], dtype=torch.float32)},
                    batch_size=len(data),
                ),
                non_tensor_batch={"acc": acc},
                meta_info={"reward_extra_keys": ["acc"]},
            )

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.reward_loop_manager = _RewardLoopManager()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                forced_answer_probe=ForcedAnswerProbeConfig(
                    enable=True,
                    training_credit=ForcedAnswerTrainingCreditConfig(enable=True),
                )
            )
        )
    )
    score = trainer._score_forced_answer_probe(
        batch=batch,
        capture=capture,
        probe_reward_batch=probe_reward_batch,
        original_reward_tensor=batch.batch["rm_scores"],
        # Balanced current order is parent 1, parent 0.
        original_reward_extra_infos={"acc": np.asarray([1.0, 0.0])},
    )

    assert score.training_credit.corrected_parent_indices == (0,)
    assert score.training_credit.effective_reward_tensor[1].sum().item() == pytest.approx(0.5)
    assert torch.equal(score.training_credit.effective_reward_tensor[0], snapshot["rm_scores"][0])
    for key, expected in snapshot.items():
        assert torch.equal(batch.batch[key], expected)
    assert "__forced_answer_probe_parent_index__" not in batch.non_tensor_batch
