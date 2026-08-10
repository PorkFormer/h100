from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.forced_answer_probe import (
    ForcedAnswerProbeCapture,
    ForcedAnswerGeneration,
    aggregate_probe_diagnostics,
    build_probe_reward_batch,
    detect_hit_response_cap,
    run_forced_answer_probe,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.config import ForcedAnswerProbeConfig


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del text, kwargs
        return {"input_ids": [90, 91]}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(int(token)) for token in token_ids)


class _Client:
    def __init__(self):
        self.calls = []

    async def generate_grouped(self, request_id, *, prompt_ids, sampling_params, routing_key=None):
        self.calls.append((request_id, prompt_ids, dict(sampling_params), routing_key))
        return [
            SimpleNamespace(token_ids=[40], extra_fields={"branch_id": 0}),
            SimpleNamespace(token_ids=[41, 42], extra_fields={"branch_id": 1}),
        ]


def _rollout_batch() -> DataProto:
    prompts = torch.tensor([[0, 11], [0, 12]], dtype=torch.long)
    responses = torch.tensor([[21, 2, 0, 0], [31, 32, 33, 34]], dtype=torch.long)
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1]], dtype=torch.long
    )
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention_mask,
            "position_ids": torch.clamp(attention_mask.cumsum(-1) - 1, min=0),
            "response_mask": attention_mask[:, -4:].clone(),
            "old_log_probs": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "advantages": torch.ones((2, 4), dtype=torch.float32),
        },
        batch_size=2,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.asarray(["eos", "cap"], dtype=object),
            "finish_reason": np.asarray(["stop", "length"], dtype=object),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray(
                [{"ground_truth": "1"}, {"ground_truth": "2"}], dtype=object
            ),
            "extra_info": np.asarray([{}, {}], dtype=object),
        },
    )


def test_cap_detection_distinguishes_eos_and_length():
    detected = detect_hit_response_cap(
        finish_reasons=["stop", "length"],
        response_lengths=[4, 4],
        max_response_length=4,
        response_token_ids=[[1, 2, 0, 0], [3, 4, 5, 6]],
        eos_token_id=2,
    )
    assert detected.tolist() == [False, True]

    fallback = detect_hit_response_cap(
        finish_reasons=[None, None],
        response_lengths=[4, 4],
        max_response_length=4,
        response_token_ids=[[1, 2, 0, 0], [3, 4, 5, 6]],
        eos_token_id=2,
    )
    assert fallback.tolist() == [False, True]


def test_probe_disabled_does_not_call_generation():
    client = _Client()
    result = run_forced_answer_probe(
        config=ForcedAnswerProbeConfig(enable=False),
        rollout_batch=_rollout_batch(),
        tokenizer=_Tokenizer(),
        client=client,
        max_response_length=4,
        global_step=3,
    )
    assert result is None
    assert client.calls == []


def test_probe_filters_to_only_hit_cap_trajectories():
    client = _Client()
    capture = run_forced_answer_probe(
        config=ForcedAnswerProbeConfig(enable=True),
        rollout_batch=_rollout_batch(),
        tokenizer=_Tokenizer(),
        client=client,
        max_response_length=4,
        global_step=3,
    )
    assert capture.hit_response_cap.tolist() == [False, True]
    assert len(client.calls) == 1
    assert {generation.parent_index for generation in capture.generations} == {1}
    assert client.calls[0][2]["n"] == 2
    assert client.calls[0][2]["max_tokens"] == 64


def test_k_sample_aggregation():
    generations = [
        ForcedAnswerGeneration(0, 0, (1,), (10,)),
        ForcedAnswerGeneration(0, 1, (1,), (11,)),
    ]
    diagnostics = aggregate_probe_diagnostics(
        hit_response_cap=[True],
        generations=generations,
        probe_correctness=[1.0, 0.0],
        original_correctness=[0.0],
        probe_shaped_rewards=[1.0, 0.0],
        original_shaped_rewards=[0.0],
        original_generated_tokens=4,
        num_samples=2,
        correctness_threshold=0.5,
        high_confidence_threshold=1.0,
    )
    assert diagnostics.metrics["probe/success_rate_mean"] == pytest.approx(0.5)
    assert diagnostics.metrics["probe/p_any_success"] == 1.0
    assert diagnostics.metrics["probe/p_all_success"] == 0.0


def test_probe_generation_and_reward_batch_do_not_mutate_training_tensors():
    original = _rollout_batch()
    tensor_snapshot = {key: value.clone() for key, value in original.batch.items()}
    client = _Client()
    capture = run_forced_answer_probe(
        config=ForcedAnswerProbeConfig(enable=True),
        rollout_batch=original,
        tokenizer=_Tokenizer(),
        client=client,
        max_response_length=4,
        global_step=5,
    )
    reward_batch = build_probe_reward_batch(original, capture.generations, pad_token_id=0)

    for key in ("responses", "response_mask", "old_log_probs", "advantages"):
        assert torch.equal(original.batch[key], tensor_snapshot[key])
    assert reward_batch.batch["responses"].shape[0] == 2
    assert reward_batch.batch["responses"].data_ptr() != original.batch["responses"].data_ptr()
    assert "probe_parent_index" not in original.non_tensor_batch


def test_logging_statistics_false_negative_and_token_overhead():
    generations = [
        ForcedAnswerGeneration(1, 0, (1,), (10, 11)),
        ForcedAnswerGeneration(1, 1, (1,), (12,)),
        ForcedAnswerGeneration(2, 0, (1,), (13,)),
        ForcedAnswerGeneration(2, 1, (1,), (14, 15)),
    ]
    diagnostics = aggregate_probe_diagnostics(
        hit_response_cap=[False, True, True],
        generations=generations,
        probe_correctness=[1.0, 0.0, 1.0, 1.0],
        original_correctness=[1.0, 0.0, 1.0],
        probe_shaped_rewards=[1.0, 0.0, 1.0, 1.0],
        original_shaped_rewards=[1.0, 0.0, 1.0],
        original_generated_tokens=12,
        num_samples=2,
        correctness_threshold=0.5,
        high_confidence_threshold=1.0,
    )
    metrics = diagnostics.metrics
    assert metrics["probe/hit_cap_rate"] == pytest.approx(2 / 3)
    assert metrics["probe/num_truncated_trajectories"] == 2.0
    assert metrics["probe/num_probe_generations"] == 4.0
    assert metrics["probe/success_rate_mean"] == pytest.approx(0.75)
    assert metrics["probe/p_any_success"] == 1.0
    assert metrics["probe/p_all_success"] == pytest.approx(0.5)
    assert metrics["probe/extra_generated_tokens"] == 6.0
    assert metrics["probe/extra_token_ratio"] == pytest.approx(0.5)
    assert metrics["probe/raw_correctness_mean"] == pytest.approx(0.75)
    assert metrics["probe/shaped_reward_mean"] == pytest.approx(0.75)
    assert metrics["probe/truncation_false_negative_candidate_rate"] == pytest.approx(0.5)
    assert metrics["probe/truncation_high_confidence_recoverable_rate"] == 0.0
    assert metrics["probe/recovery_rate_given_truncated_failure"] == 1.0


def test_shaped_reward_does_not_contaminate_correctness():
    generations = [
        ForcedAnswerGeneration(0, 0, (1,), (10,)),
        ForcedAnswerGeneration(0, 1, (1,), (11,)),
    ]
    diagnostics = aggregate_probe_diagnostics(
        hit_response_cap=[True],
        generations=generations,
        probe_correctness=[1.0, 1.0],
        original_correctness=[1.0],
        probe_shaped_rewards=[1.0, 1.0],
        original_shaped_rewards=[-0.2],
        original_generated_tokens=4,
        num_samples=2,
        correctness_threshold=0.5,
        high_confidence_threshold=1.0,
    )
    assert diagnostics.metrics["probe/truncation_false_negative_candidate_rate"] == 0.0
    assert diagnostics.metrics["probe/recovery_rate_given_truncated_failure"] == 0.0


@pytest.mark.parametrize(
    ("probe_correctness", "expected_all", "expected_high_confidence"),
    [([1.0, 0.0], 0.0, 0.0), ([1.0, 1.0], 1.0, 1.0)],
)
def test_raw_correctness_controls_recovery_and_high_confidence(
    probe_correctness, expected_all, expected_high_confidence
):
    generations = [
        ForcedAnswerGeneration(0, 0, (1,), (10,)),
        ForcedAnswerGeneration(0, 1, (1,), (11,)),
    ]
    diagnostics = aggregate_probe_diagnostics(
        hit_response_cap=[True],
        generations=generations,
        probe_correctness=probe_correctness,
        original_correctness=[0.0],
        probe_shaped_rewards=[-10.0, -10.0],
        original_shaped_rewards=[10.0],
        original_generated_tokens=4,
        num_samples=2,
        correctness_threshold=0.5,
        high_confidence_threshold=1.0,
    )
    metrics = diagnostics.metrics
    assert metrics["probe/p_any_success"] == 1.0
    assert metrics["probe/p_all_success"] == expected_all
    assert metrics["probe/truncation_false_negative_candidate_rate"] == 1.0
    assert metrics["probe/truncation_high_confidence_recoverable_rate"] == expected_high_confidence


def test_conditional_recovery_excludes_originally_correct_trajectories():
    generations = [
        ForcedAnswerGeneration(parent, branch, (1,), (10 + branch,))
        for parent in range(4)
        for branch in range(2)
    ]
    diagnostics = aggregate_probe_diagnostics(
        hit_response_cap=[True] * 4,
        generations=generations,
        probe_correctness=[1, 0, 1, 0, 0, 0, 1, 1],
        original_correctness=[0, 0, 0, 1],
        probe_shaped_rewards=[0.0] * 8,
        original_shaped_rewards=[0.0] * 4,
        original_generated_tokens=16,
        num_samples=2,
        correctness_threshold=0.5,
        high_confidence_threshold=1.0,
    )
    assert diagnostics.metrics["probe/recovery_rate_given_truncated_failure"] == pytest.approx(2 / 3)


def test_probe_reward_scoring_pads_only_the_independent_reward_batch():
    original = _rollout_batch()
    original.batch["rm_scores"] = torch.zeros_like(original.batch["responses"], dtype=torch.float32)
    original.non_tensor_batch["__forced_answer_probe_parent_index__"] = np.arange(2, dtype=np.int64)
    tensor_snapshot = {key: value.clone() for key, value in original.batch.items()}
    generations = (
        ForcedAnswerGeneration(1, 0, (1,), (40,)),
        ForcedAnswerGeneration(1, 1, (1,), (41,)),
    )
    capture = ForcedAnswerProbeCapture(
        hit_response_cap=np.asarray([False, True]),
        generations=generations,
    )
    reward_batch = build_probe_reward_batch(original, generations, pad_token_id=0)

    class _RewardLoopManager:
        reward_loop_workers = [object(), object(), object(), object()]

        def __init__(self):
            self.observed_batch_sizes = []

        def compute_rm_score(self, data):
            self.observed_batch_sizes.append(len(data))
            assert len(data) % len(self.reward_loop_workers) == 0
            rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            branch_ids = torch.tensor(data.non_tensor_batch["probe_branch_id"])
            rm_scores[:, 0] = (branch_ids == 0).float()
            return DataProto(
                batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(data)),
                non_tensor_batch={"acc": (branch_ids == 0).numpy().astype(np.float32)},
                meta_info={"reward_extra_keys": ["acc"]},
            )

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.reward_loop_manager = _RewardLoopManager()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(forced_answer_probe=ForcedAnswerProbeConfig(enable=True))
        )
    )
    metrics = trainer._score_forced_answer_probe(
        batch=original,
        capture=capture,
        probe_reward_batch=reward_batch,
        original_reward_tensor=original.batch["rm_scores"],
        original_reward_extra_infos={"acc": np.asarray([0.0, 0.0])},
    )

    assert trainer.reward_loop_manager.observed_batch_sizes == [4]
    assert metrics["probe/num_probe_generations"] == 2.0
    assert metrics["probe/success_rate_mean"] == pytest.approx(0.5)
    for key, snapshot in tensor_snapshot.items():
        assert torch.equal(original.batch[key], snapshot)


def test_missing_correctness_key_fails_closed():
    original = _rollout_batch()
    original.batch["rm_scores"] = torch.zeros_like(original.batch["responses"], dtype=torch.float32)
    original.non_tensor_batch["__forced_answer_probe_parent_index__"] = np.arange(2, dtype=np.int64)
    generations = (
        ForcedAnswerGeneration(1, 0, (1,), (40,)),
        ForcedAnswerGeneration(1, 1, (1,), (41,)),
    )
    capture = ForcedAnswerProbeCapture(
        hit_response_cap=np.asarray([False, True]),
        generations=generations,
    )
    reward_batch = build_probe_reward_batch(original, generations, pad_token_id=0)

    class _RewardLoopManager:
        reward_loop_workers = [object()]

        def compute_rm_score(self, data):
            rm_scores = torch.ones_like(data.batch["responses"], dtype=torch.float32)
            return DataProto(batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(data)))

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.reward_loop_manager = _RewardLoopManager()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(forced_answer_probe=ForcedAnswerProbeConfig(enable=True))
        )
    )
    with pytest.raises(RuntimeError, match="requires raw verifier correctness field 'acc'"):
        trainer._score_forced_answer_probe(
            batch=original,
            capture=capture,
            probe_reward_batch=reward_batch,
            original_reward_tensor=original.batch["rm_scores"],
            original_reward_extra_infos={"acc": np.asarray([0.0, 0.0])},
        )


def test_forced_answer_probe_config_correctness_defaults_and_validation():
    config = ForcedAnswerProbeConfig()
    assert config.correctness_key == "acc"
    assert config.correctness_threshold == 0.5
    assert config.high_confidence_threshold == 1.0
    with pytest.raises(ValueError, match="correctness_key"):
        ForcedAnswerProbeConfig(correctness_key="").validate()
    with pytest.raises(ValueError, match="correctness_threshold"):
        ForcedAnswerProbeConfig(correctness_threshold=float("nan")).validate()
