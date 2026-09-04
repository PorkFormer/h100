# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import copy
import json
import random
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.natural_continuation_boundary_return.accumulator import (
    BoundaryReturnStepAccumulator,
)
from verl.experimental.natural_continuation_boundary_return.dapo_trainer import (
    RayDAPOBoundaryReturnTrainer,
    validate_boundary_return_preflight,
)
from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BoundaryReturnBatchResult,
)
from verl.experimental.natural_continuation_boundary_return.runtime import BoundaryContinuationGeneration
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer
from verl.trainer.config import ProbeCreditConfig
from verl.workers.config.rollout import BoundaryReturnConfig, ForcedAnswerProbeConfig


def test_baseline_calibration_exports_only_exact_cap_prefix_rows(tmp_path):
    trainer = RayDAPOBoundaryReturnTrainer.__new__(RayDAPOBoundaryReturnTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(hard_prefix_source_path=str(tmp_path / "cap-source.jsonl")),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(response_length=4)),
    )
    trainer._typed_boundary_return_config = BoundaryReturnConfig(mode="off")
    trainer._rollout_policy_version = 0
    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2], [3, 4]]),
                "responses": torch.tensor([[5, 6, 7, 8], [9, 10, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
                "response_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "finish_reason": np.asarray(["length", "stop"], dtype=object),
            "trajectory_id": np.asarray(["t0", "t1"], dtype=object),
            "rollout_policy_version": np.asarray([0, 0], dtype=object),
            "prompt_id": np.asarray(["p0", "p1"], dtype=object),
            "data_source": np.asarray(["a", "b"], dtype=object),
        },
    )
    trainer._after_normal_rollout(batch)
    rows = [json.loads(line) for line in (tmp_path / "cap-source.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "p0"
    assert rows[0]["response_token_ids"] == [5, 6, 7, 8]
    assert rows[0]["source_context"]["data_source"] == "a"


def test_candidate_mechanism_rows_keep_newly_locked_outside_actor_labels(tmp_path):
    trainer = RayDAPOBoundaryReturnTrainer.__new__(RayDAPOBoundaryReturnTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(mechanism_rows_path=str(tmp_path / "rows.jsonl")))
    trainer._typed_boundary_return_config = BoundaryReturnConfig(mode="replace", correctness_threshold=0.5)
    trainer.global_steps = 2
    candidate = DataProto(
        batch=TensorDict({}, batch_size=4),
        non_tensor_batch={
            "trajectory_id": np.asarray(["t0", "t1", "t2", "t3"], dtype=object),
        },
    )
    result = SimpleNamespace(
        uids=np.asarray(["newly-locked", "newly-locked", "unchanged", "unchanged"], dtype=object),
        short_acc=np.asarray([0.0, 1.0, 0.0, 0.0]),
        long_acc=np.asarray([0.0, 0.0, np.nan, np.nan]),
        boundary_acc=np.asarray([0.0, 0.0, 0.0, 0.0]),
        task_score_delta=np.asarray([0.0, -1.0, 0.0, 0.0]),
        tail_token_lengths=np.asarray([10, 20, 0, 0]),
        long_hit_response_cap=np.asarray([False, False, False, False]),
        hit_response_cap=np.asarray([True, True, False, False]),
    )
    trainer._write_candidate_mechanism_rows(candidate, result, generation_batch_index=1)
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["boundary_group_newly_locked"] for row in rows)
    assert "boundary_group_newly_locked" not in candidate.batch


def _result(
    *,
    uids,
    short,
    long,
    boundary,
    tails,
    normal_tokens,
    short_task=None,
    long_task=None,
) -> BoundaryReturnBatchResult:
    short = np.asarray(short, dtype=np.float64)
    long = np.asarray(long, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    hit = np.isfinite(long)
    short_task = short if short_task is None else np.asarray(short_task, dtype=np.float64)
    long_task = long if long_task is None else np.asarray(long_task, dtype=np.float64)
    boundary_task = short_task.copy()
    boundary_task[hit] = long_task[hit]
    return BoundaryReturnBatchResult(
        hit_response_cap=hit,
        short_acc=short,
        long_acc=long,
        boundary_acc=boundary,
        short_task_score=short_task,
        long_task_score=long_task,
        boundary_task_score=boundary_task,
        task_score_delta=boundary_task - short_task,
        tail_token_lengths=np.asarray(tails, dtype=np.int64),
        normal_response_tokens=normal_tokens,
        uids=np.asarray(uids, dtype=object),
        metrics={"boundary_return/prefix_penalty_drift_max": 0.0},
    )


def test_step_accumulator_uses_all_rows_and_exact_global_denominators_and_quantiles():
    accumulator = BoundaryReturnStepAccumulator(correctness_threshold=0.5)
    accumulator.add(
        _result(
            uids=["all-wrong-unlocked"] * 2 + ["kept"] * 2,
            short=[0, 0, 0, 1],
            long=[1, np.nan, 1, 0],
            boundary=[1, 0, 0, 0],
            tails=[1, 0, 3, 2],
            normal_tokens=8,
            short_task=[0, 0, 0, 1],
            long_task=[1, np.nan, 0, -1],
        )
    )
    accumulator.add(
        _result(
            uids=["all-wrong-still-filtered"] * 2,
            short=[0, 0],
            long=[0, 1],
            boundary=[0, 1],
            tails=[4, 5],
            normal_tokens=4,
        )
    )

    metrics = accumulator.metrics()
    # All tails [1, 3, 2, 4, 5] are pooled; no mean-of-batch-quantiles.
    assert metrics["boundary_return/extra_generated_tokens"] == 15.0
    assert metrics["boundary_return/extra_generated_token_ratio"] == 15 / 12
    assert metrics["boundary_return/tail_tokens_p50"] == 3.0
    assert metrics["boundary_return/tail_tokens_p90"] == pytest.approx(4.6)
    assert metrics["boundary_return/recovered_rate_given_cap_failure"] == 3 / 4
    assert metrics["boundary_return/long_success_rate_given_cap"] == 3 / 5
    assert metrics["boundary_return/regressed_count"] == 1.0
    assert metrics["boundary_return/regressed_rate_given_cap_success"] == 1.0
    assert metrics["boundary_return/short_all_wrong_group_count"] == 2.0
    assert metrics["boundary_return/unlocked_group_count"] == 2.0
    assert metrics["boundary_return/unlocked_group_rate"] == 1.0
    assert metrics["boundary_return/newly_locked_group_count"] == 1.0
    assert metrics["boundary_return/newly_locked_group_rate"] == 1.0
    assert metrics["boundary_return/prefix_penalty_drift_max"] == 0.0


def test_group_std_transitions_include_all_correct_groups_not_only_all_wrong_groups():
    accumulator = BoundaryReturnStepAccumulator(correctness_threshold=0.5)
    accumulator.add(
        _result(
            uids=["all-correct"] * 2 + ["variable"] * 2,
            short=[1, 1, 0, 1],
            long=[1, 0, 0, 0],
            boundary=[1, 0, 0, 0],
            tails=[1, 1, 1, 1],
            normal_tokens=16,
        )
    )

    metrics = accumulator.metrics()
    assert metrics["boundary_return/short_all_wrong_group_count"] == 0.0
    assert metrics["boundary_return/unlocked_group_count"] == 1.0
    assert metrics["boundary_return/newly_locked_group_count"] == 1.0


def _config(mode="replace"):
    boundary = BoundaryReturnConfig(
        mode=mode,
        long_response_length=8,
        max_concurrent_requests=2,
        request_batch_size=2,
    )
    algorithm = SimpleNamespace(
        adv_estimator="grpo",
        use_kl_in_reward=False,
        rollout_correction=None,
        filter_groups=SimpleNamespace(enable=True, metric="acc", max_num_gen_batches=3),
        probe_credit=ProbeCreditConfig(enable=False, coef=0.0),
        censor_aware_advantage=SimpleNamespace(enable=False),
        readiness_dominance=SimpleNamespace(mode="off"),
        success_support_floor=SimpleNamespace(mode="off"),
        on_policy_budgeted_capability_floor=SimpleNamespace(mode="off"),
    )
    algorithm.get = lambda name, default=None: getattr(algorithm, name, default)
    rollout = SimpleNamespace(
        name="vllm",
        mode="async",
        response_length=4,
        prompt_length=2,
        max_model_len=10,
        ignore_eos=False,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        calculate_log_probs=False,
        n=2,
        multi_turn=SimpleNamespace(enable=False),
        agent=SimpleNamespace(
            default_agent_loop="single_turn_agent",
            agent_loop_config_path=None,
            agent_loop_manager_class=None,
            custom_async_server=SimpleNamespace(path=None, name=None),
        ),
        forced_answer_probe=ForcedAnswerProbeConfig(enable=False),
        boundary_return=boundary,
    )
    config = SimpleNamespace(
        algorithm=algorithm,
        actor_rollout_ref=SimpleNamespace(rollout=rollout),
        critic=SimpleNamespace(enable=False),
        reward=SimpleNamespace(
            reward_manager=SimpleNamespace(source="register", name="dapo"),
            custom_reward_function=SimpleNamespace(path=None),
            reward_model=SimpleNamespace(enable=False),
            sandbox_fusion=SimpleNamespace(url=None),
        ),
        distillation=SimpleNamespace(enabled=False),
        global_profiler=SimpleNamespace(steps=None),
    )
    return config


def _trainer(mode="replace"):
    trainer = object.__new__(RayDAPOBoundaryReturnTrainer)
    trainer.config = _config(mode)
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    return trainer


def test_base_dapo_exposes_exactly_two_default_noop_candidate_filter_hooks():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config("off")
    candidate = DataProto.from_dict(tensors={"x": torch.ones(1, 1)})
    assert trainer._process_candidate_after_reward_before_filter(candidate, {}, {}, 1) is candidate
    assert trainer._effective_filter_metric() == "acc"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: setattr(t.config.actor_rollout_ref.rollout, "name", "sglang"), "vLLM"),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout, "mode", "sync"), "async"),
        (lambda t: setattr(t.config.algorithm, "adv_estimator", "gae"), "GRPO"),
        (lambda t: setattr(t, "use_critic", True), "no critic"),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout, "ignore_eos", True), "ignore_eos"),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout.multi_turn, "enable", True), "single-turn"),
        (lambda t: setattr(t.config.algorithm, "use_kl_in_reward", True), "use_kl_in_reward"),
        (
            lambda t: setattr(t.config.actor_rollout_ref.rollout.agent, "default_agent_loop", "tool_agent"),
            "single_turn_agent",
        ),
        (
            lambda t: setattr(t.config.actor_rollout_ref.rollout.agent, "agent_loop_config_path", "custom.yaml"),
            "custom agent",
        ),
        (
            lambda t: setattr(t.config.actor_rollout_ref.rollout.agent.custom_async_server, "path", "server.py"),
            "custom async server",
        ),
        (lambda t: setattr(t.config.reward.reward_manager, "name", "naive"), "DAPO reward manager"),
        (lambda t: setattr(t.config.reward.reward_manager, "source", "importlib"), "registered DAPO"),
        (lambda t: setattr(t.config.reward.custom_reward_function, "path", "reward.py"), "custom reward"),
        (lambda t: setattr(t.config.reward.reward_model, "enable", True), "reward model"),
        (lambda t: setattr(t.config.reward.sandbox_fusion, "url", "http://sandbox"), "sandbox"),
        (lambda t: setattr(t.config.distillation, "enabled", True), "distillation"),
        (
            lambda t: setattr(
                t.config.algorithm,
                "rollout_correction",
                SimpleNamespace(rollout_is=SimpleNamespace(), rollout_rs=None, bypass_mode=False),
            ),
            "rollout correction",
        ),
        (lambda t: setattr(t.config.global_profiler, "steps", [1]), "profiling steps"),
        (
            lambda t: setattr(
                t.config.actor_rollout_ref.rollout,
                "boundary_return",
                replace(t.config.actor_rollout_ref.rollout.boundary_return, correctness_key="ok"),
            ),
            "correctness_key=acc",
        ),
        (
            lambda t: setattr(
                t.config.actor_rollout_ref.rollout,
                "boundary_return",
                replace(t.config.actor_rollout_ref.rollout.boundary_return, task_score_key="task"),
            ),
            "task_score_key=score",
        ),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout, "response_length", 8), "L > H"),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout, "max_model_len", 9), "context"),
        (lambda t: setattr(t.config.algorithm.filter_groups, "enable", False), "filter_groups.enable"),
        (lambda t: setattr(t.config.algorithm.filter_groups, "metric", "score"), "Hydra filter_groups.metric"),
        (
            lambda t: setattr(
                t.config.actor_rollout_ref.rollout,
                "forced_answer_probe",
                SimpleNamespace(enable=True, training_credit=SimpleNamespace(enable=False)),
            ),
            "forced-answer",
        ),
        (lambda t: setattr(t.config.algorithm.censor_aware_advantage, "enable", True), "FA-CAC/FA-RAR"),
        (
            lambda t: setattr(t.config.algorithm, "probe_credit", ProbeCreditConfig(enable=True, coef=1.0)),
            "Probe Credit",
        ),
        (lambda t: setattr(t.config.algorithm.readiness_dominance, "mode", "shadow"), "Readiness"),
        (lambda t: setattr(t.config.algorithm.success_support_floor, "mode", "shadow"), "BSSF"),
        (
            lambda t: setattr(t.config.algorithm.on_policy_budgeted_capability_floor, "mode", "shadow"),
            "OBCF",
        ),
    ],
)
def test_active_boundary_return_rejects_unverified_modes_and_interventions(mutate, message):
    trainer = _trainer()
    mutate(trainer)
    with pytest.raises(ValueError, match=message):
        trainer._validate_probe_credit_mode()


def test_pure_preflight_rejects_before_runtime_objects_exist():
    config = _config("replace")
    validate_boundary_return_preflight(config)
    config.reward.reward_manager.name = "naive"
    with pytest.raises(ValueError, match="DAPO reward manager"):
        validate_boundary_return_preflight(config)


def test_candidate_hook_rejects_actual_multimodal_rows_before_continuation():
    trainer = _trainer("shadow")
    trainer._rollout_policy_version = 7
    candidate = _hook_candidate()
    candidate.non_tensor_batch["multi_modal_data"] = np.asarray([{"image": object()}, None], dtype=object)
    with pytest.raises(ValueError, match="multimodal"):
        trainer._process_candidate_after_reward_before_filter(candidate, {}, {}, 1)


def test_candidate_hook_rejects_actual_non_single_turn_agent_rows():
    trainer = _trainer("shadow")
    trainer._rollout_policy_version = 7
    candidate = _hook_candidate()
    candidate.non_tensor_batch["agent_name"] = np.asarray(["single_turn_agent", "tool_agent"], dtype=object)
    with pytest.raises(ValueError, match="every row.*single_turn_agent"):
        trainer._process_candidate_after_reward_before_filter(candidate, {}, {}, 1)


def test_shadow_does_not_require_filter_but_replace_uses_local_metric_without_mutating_hydra():
    shadow = _trainer("shadow")
    shadow.config.algorithm.filter_groups.enable = False
    shadow._validate_probe_credit_mode()
    assert shadow._effective_filter_metric() is None

    replace = _trainer("replace")
    replace._validate_probe_credit_mode()
    assert replace.config.algorithm.filter_groups.metric == "acc"
    assert replace._effective_filter_metric() == "boundary_acc"
    assert replace.config.algorithm.filter_groups.metric == "acc"


def _hook_candidate() -> DataProto:
    prompts = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    responses = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=torch.long)
    mask = torch.ones((2, 6), dtype=torch.long)
    scores = torch.zeros((2, 4), dtype=torch.float32)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), -1),
            "attention_mask": mask,
            "position_ids": torch.arange(6).repeat(2, 1),
            "response_mask": torch.ones((2, 4), dtype=torch.long),
            "token_level_scores": scores.clone(),
            "token_level_rewards": scores.clone(),
        },
        non_tensors={
            "uid": np.asarray(["u", "u"], dtype=object),
            "trajectory_id": np.asarray(["u:0", "u:1"], dtype=object),
            "rollout_policy_version": np.asarray([7, 7], dtype=object),
            "finish_reason": np.asarray(["length", "length"], dtype=object),
            "acc": np.asarray([0.0, 1.0]),
            "score": np.asarray([0.0, 1.0]),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "0"}] * 2, dtype=object),
        },
        meta_info={"reward_extra_keys": ["acc", "score"]},
    )


def test_candidate_hook_orders_continuation_long_reward_replacement_and_preserves_rng(monkeypatch):
    from verl.experimental.natural_continuation_boundary_return import dapo_trainer as module

    trainer = _trainer("replace")
    trainer._rollout_policy_version = 7
    trainer.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    trainer.llm_server_manager = SimpleNamespace(get_client=lambda: object())
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: pytest.fail("must not sleep"))
    events = []
    capture = SimpleNamespace(hit_response_cap=np.asarray([True, True]), generations=(object(), object()))

    def continuation(**_kwargs):
        events.append("continuation")
        random.random()
        np.random.random()
        torch.rand(1)
        return capture

    long_batch = DataProto.from_dict(tensors={"responses": torch.ones(2, 1, dtype=torch.long)})
    monkeypatch.setattr(module, "run_boundary_continuations", continuation)
    monkeypatch.setattr(
        module,
        "build_long_reward_batch",
        lambda *_args, **_kwargs: events.append("build") or long_batch,
    )
    trainer._score_batch_with_existing_reward_pipeline = MethodType(
        lambda self, batch: (
            events.append("long_reward")
            or SimpleNamespace(reward_tensor=torch.ones(2, 1), extra_info={"acc": [0, 1], "score": [0, 1]})
        ),
        trainer,
    )
    result = _result(uids=["u", "u"], short=[0, 1], long=[0, 1], boundary=[0, 1], tails=[0, 1], normal_tokens=8)

    def apply(candidate, **_kwargs):
        events.append("replace")
        candidate.non_tensor_batch["boundary_acc"] = np.asarray([0.0, 1.0])
        return result

    monkeypatch.setattr(module, "apply_boundary_return", apply)
    py_before = random.getstate()
    np_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    candidate = _hook_candidate()

    returned = trainer._process_candidate_after_reward_before_filter(candidate, {}, {}, 1)

    assert returned is candidate
    assert events == ["continuation", "build", "long_reward", "replace"]
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0] and np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_long_reward_scoring_chunks_6144_rows_and_keeps_only_scalar_outputs(monkeypatch):
    from verl.experimental.natural_continuation_boundary_return import dapo_trainer as module

    trainer = _trainer("shadow")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    chunk_sizes = []

    def build(_candidate, generations, **_kwargs):
        chunk_sizes.append(len(generations))
        return DataProto.from_dict(tensors={"responses": torch.ones(len(generations), 1)})

    monkeypatch.setattr(module, "build_long_reward_batch", build)

    def score(_self, batch):
        size = len(batch)
        return SimpleNamespace(
            reward_tensor=torch.full((size, 8192), 1.0e30),
            extra_info={"acc": np.ones(size), "score": np.arange(size, dtype=np.float64)},
        )

    trainer._score_batch_with_existing_reward_pipeline = MethodType(score, trainer)
    generations = tuple(
        BoundaryContinuationGeneration(
            parent_index=row,
            request_id=f"r-{row}",
            branch_id=0,
            uid=f"u-{row}",
            trajectory_id=f"t-{row}",
            prompt_token_ids=(1,),
            prefix_token_ids=(2,),
            tail_token_ids=(3,),
            actual_policy_version=7,
        )
        for row in range(6144)
    )

    output = trainer._score_long_generations_in_chunks(
        DataProto.from_dict(tensors={"responses": torch.ones(6144, 1)}),
        generations,
        BoundaryReturnConfig(mode="shadow", long_response_length=8, long_reward_chunk_size=256),
    )

    assert len(chunk_sizes) == 24
    assert set(chunk_sizes) == {256}
    assert output.reward_tensor.shape == (6144, 0)
    assert set(output.extra_info) == {"acc", "score"}
    assert len(output.extra_info["acc"]) == 6144


def test_long_reward_scoring_pads_nondivisible_chunk_only_inside_reward_batch(monkeypatch):
    from verl.experimental.natural_continuation_boundary_return import dapo_trainer as module

    trainer = _trainer("shadow")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_workers=[object()] * 8)
    candidate = DataProto.from_dict(tensors={"responses": torch.arange(12).reshape(12, 1)})
    built_sizes = []
    scored_sizes = []

    def build(_candidate, generations, **_kwargs):
        built_sizes.append(len(generations))
        return DataProto.from_dict(tensors={"responses": torch.arange(len(generations)).reshape(-1, 1)})

    monkeypatch.setattr(module, "build_long_reward_batch", build)

    def score(_self, batch):
        scored_sizes.append(len(batch))
        assert len(batch) % 8 == 0
        return SimpleNamespace(
            reward_tensor=torch.full((len(batch), 1), 1.0e30),
            extra_info={
                "acc": np.arange(len(batch), dtype=np.float64),
                "score": np.arange(len(batch), dtype=np.float64) + 100,
            },
        )

    trainer._score_batch_with_existing_reward_pipeline = MethodType(score, trainer)
    generations = tuple(
        BoundaryContinuationGeneration(
            parent_index=row,
            request_id=f"r-{row}",
            branch_id=0,
            uid=f"u-{row}",
            trajectory_id=f"t-{row}",
            prompt_token_ids=(1,),
            prefix_token_ids=(2,),
            tail_token_ids=(3,),
            actual_policy_version=7,
        )
        for row in range(12)
    )

    output = trainer._score_long_generations_in_chunks(
        candidate,
        generations,
        BoundaryReturnConfig(mode="shadow", long_response_length=8, long_reward_chunk_size=256),
    )

    assert built_sizes == [12]
    assert scored_sizes == [16]
    assert len(candidate) == 12
    assert output.extra_info["acc"].tolist() == list(range(12))
    assert output.extra_info["score"].tolist() == [100 + row for row in range(12)]


@pytest.mark.parametrize("failure_stage", ["continuation", "long_reward"])
def test_candidate_failure_sleeps_exactly_once_and_reraises(monkeypatch, failure_stage):
    from verl.experimental.natural_continuation_boundary_return import dapo_trainer as module

    trainer = _trainer("replace")
    trainer._rollout_policy_version = 7
    trainer.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    trainer.llm_server_manager = SimpleNamespace(get_client=lambda: object())
    sleeps = []
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: sleeps.append("sleep"))
    capture = SimpleNamespace(hit_response_cap=np.asarray([True, True]), generations=(object(), object()))

    def continuation(**_kwargs):
        if failure_stage == "continuation":
            raise RuntimeError("continuation failed")
        return capture

    monkeypatch.setattr(module, "run_boundary_continuations", continuation)
    monkeypatch.setattr(
        module,
        "build_long_reward_batch",
        lambda *_args, **_kwargs: DataProto.from_dict(tensors={"responses": torch.ones(2, 1)}),
    )

    def score(_self, _batch):
        raise RuntimeError("long reward failed")

    trainer._score_batch_with_existing_reward_pipeline = MethodType(score, trainer)
    with pytest.raises(RuntimeError, match=failure_stage.replace("_", " ")):
        trainer._process_candidate_after_reward_before_filter(_hook_candidate(), {}, {}, 1)
    assert sleeps == ["sleep"]


def test_candidate_failure_does_not_sleep_when_remote_drain_release_is_unattested(monkeypatch):
    from verl.experimental.natural_continuation_boundary_return import dapo_trainer as module

    trainer = _trainer("replace")
    trainer._rollout_policy_version = 7
    trainer.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    trainer.llm_server_manager = SimpleNamespace(get_client=lambda: object())
    sleeps = []
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: sleeps.append("sleep"))

    def continuation(**_kwargs):
        error = RuntimeError("continuation failed")
        error.boundary_remote_cleanup_attested = False
        raise error

    monkeypatch.setattr(module, "run_boundary_continuations", continuation)
    with pytest.raises(RuntimeError, match="continuation failed") as caught:
        trainer._process_candidate_after_reward_before_filter(_hook_candidate(), {}, {}, 1)
    assert sleeps == []
    assert any("not slept" in note for note in caught.value.__notes__)


def test_trainer_sleeps_only_after_runtime_remote_drain_and_release():
    trainer = _trainer("replace")
    trainer._rollout_policy_version = 7
    trainer.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    events = []
    both_started = asyncio.Event()

    class Tracked:
        server_id = "server"
        server = object()
        object_ref = object()

        def __init__(self, request_id, fail):
            self.request_id = request_id
            self.backend_request_id = request_id
            self.fail = fail

        async def result(self):
            events.append("result")
            if events.count("result") == 2:
                both_started.set()
            await both_started.wait()
            if self.fail:
                raise RuntimeError("remote boom")
            await asyncio.sleep(60)

        async def abort(self):
            events.append("abort")

        async def drain(self):
            events.append("drain")

        async def release(self):
            events.append("release")

    class Client:
        def __init__(self):
            self.count = 0

        async def start_grouped(self, request_id, **_kwargs):
            self.count += 1
            return Tracked(request_id, self.count == 1)

    trainer.llm_server_manager = SimpleNamespace(get_client=lambda: Client())
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append("sleep"))

    with pytest.raises(RuntimeError, match="remote boom"):
        trainer._process_candidate_after_reward_before_filter(_hook_candidate(), {}, {}, 1)

    abort_positions = [index for index, event in enumerate(events) if event == "abort"]
    drain_position = events.index("drain")
    release_positions = [index for index, event in enumerate(events) if event == "release"]
    sleep_position = events.index("sleep")
    assert max(abort_positions) < drain_position < min(release_positions) < sleep_position


def _run_fit_harness(
    monkeypatch,
    *,
    mode,
    generation_batches=2,
    failure_stage=None,
    gate_cycles=0,
    gate_receipt_path=None,
):
    from verl.experimental.natural_continuation_boundary_return import dapo_trainer as boundary_module
    from verl.experimental.probe_credit import dapo_trainer as base_module
    from verl.experimental.probe_credit.dynamic_sampling import (
        filter_dapo_generation_batch as real_filter,
    )

    class Progress:
        def update(self, _count):
            pass

        def close(self):
            pass

    class Logger:
        def __init__(self):
            self.records = []

        def log(self, data, step):
            self.records.append((dict(data), step))

    events = []
    filter_spy = []
    logger = Logger()
    monkeypatch.setattr(base_module, "tqdm", lambda **_kwargs: Progress())
    monkeypatch.setattr(base_module, "Tracking", lambda **_kwargs: logger)
    monkeypatch.setattr(base_module, "compute_response_mask", lambda batch: batch.batch["attention_mask"][:, -4:])
    monkeypatch.setattr(base_module, "compute_data_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(base_module, "compute_timing_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(base_module, "compute_throughout_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(base_module, "should_save_ckpt_esi", lambda **_kwargs: False)

    def filter_spied(candidate, metric):
        events.append("filter")
        filter_spy.append(
            (
                metric,
                set(candidate.non_tensor_batch),
                candidate.non_tensor_batch["uid"].tolist(),
            )
        )
        return real_filter(candidate, metric)

    monkeypatch.setattr(base_module, "filter_dapo_generation_batch", filter_spied)
    original_continuation = boundary_module.run_boundary_continuations

    def continuation_spied(**kwargs):
        events.append(f"continuation({kwargs['policy_version']})")
        if failure_stage == "continuation":
            raise RuntimeError("continuation failed")
        return original_continuation(**kwargs)

    monkeypatch.setattr(boundary_module, "run_boundary_continuations", continuation_spied)

    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "rollout_correction": None,
                "filter_groups": {"enable": True, "metric": "acc", "max_num_gen_batches": 3},
                "probe_credit": {"enable": False, "coef": 0.0},
                "censor_aware_advantage": {"enable": False},
                "readiness_dominance": {"mode": "off"},
                "success_support_floor": {"mode": "off"},
                "on_policy_budgeted_capability_floor": {"mode": "off"},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "mode": "async",
                    "n": 2,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                    "calculate_log_probs": False,
                    "prompt_length": 2,
                    "response_length": 4,
                    "max_model_len": 10,
                    "ignore_eos": False,
                    "multi_turn": {"enable": False},
                    "agent": {
                        "default_agent_loop": "single_turn_agent",
                        "agent_loop_config_path": None,
                        "agent_loop_manager_class": None,
                        "custom_async_server": {"path": None, "name": None},
                    },
                    "forced_answer_probe": {"enable": False, "training_credit": {"enable": False}},
                    "boundary_return": {
                        "mode": mode,
                        "long_response_length": 8,
                        "correctness_key": "acc",
                        "correctness_threshold": 0.5,
                        "task_score_key": "score",
                        "max_concurrent_requests": 4,
                        "request_batch_size": 8,
                        "request_timeout_seconds": 600.0,
                        "long_reward_chunk_size": 256,
                        "verify_shadow_candidate_noop": mode == "shadow",
                        "seed": 3,
                        "strict": True,
                    },
                }
            },
            "data": {"train_batch_size": generation_batches},
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": ["console"],
                "total_epochs": 16,
                "val_before_train": False,
                "val_only": False,
                "balance_batch": False,
                "save_freq": 0,
                "test_freq": 0,
                "rollout_data_dir": None,
                "esi_redundant_time": 0,
                "dynamic_sampling_gate_cycles": gate_cycles,
                "dynamic_sampling_gate_receipt_path": gate_receipt_path,
                "profile_interval_path": None,
            },
            "distillation": {"enabled": False},
            "global_profiler": {"steps": None},
            "reward": {
                "reward_manager": {"source": "register", "name": "dapo"},
                "custom_reward_function": {"path": None},
                "reward_model": {"enable": False},
                "sandbox_fusion": {"url": None},
            },
            "critic": {"enable": False},
        }
    )
    trainer = object.__new__(RayDAPOBoundaryReturnTrainer)
    trainer.config = config
    trainer.total_training_steps = 8
    trainer.train_dataloader = [
        {
            "marker": torch.tensor([[batch_index * 2], [batch_index * 2 + 1]]),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "0"}] * 2, dtype=object),
            "extra_info": np.asarray([{"batch": batch_index}] * 2, dtype=object),
        }
        for batch_index in range(generation_batches)
    ]
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._dump_futures = []
    trainer._init_dump_executor = MethodType(lambda self: None, trainer)
    trainer._shutdown_dump_executor = MethodType(lambda self: events.append("shutdown"), trainer)
    trainer._load_checkpoint = MethodType(lambda self: setattr(self, "global_steps", 7), trainer)
    trainer._get_gen_batch = MethodType(lambda self, batch: batch, trainer)
    trainer._capture_nondeterminism_boundary = MethodType(lambda self, *_args, **_kwargs: None, trainer)
    trainer._dump_gate_equivalence_batch = MethodType(lambda self, _batch: None, trainer)
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)

    uid_values = iter([f"uid-{index}" for index in range(max(generation_batches, gate_cycles) * 2)])
    monkeypatch.setattr(base_module.uuid, "uuid4", lambda: next(uid_values))

    class RolloutManager:
        def __init__(self):
            self.calls = 0

        def generate_sequences(self, gen_input):
            assert awake["value"]
            call = self.calls
            self.calls += 1
            events.append("normal(7)")
            size = len(gen_input)
            prompts = torch.tensor([[10, 11]] * size, dtype=torch.long)
            responses = torch.tensor([[20 + call] * 4] * size, dtype=torch.long)
            attention = torch.ones((size, 6), dtype=torch.long)
            return DataProto.from_dict(
                tensors={
                    "prompts": prompts,
                    "responses": responses,
                    "input_ids": torch.cat((prompts, responses), -1),
                    "attention_mask": attention,
                    "position_ids": torch.arange(6).repeat(size, 1),
                },
                non_tensors={
                    "global_steps": np.asarray([7] * size, dtype=object),
                    "finish_reason": np.asarray(["length"] * size, dtype=object),
                },
                meta_info={"timing": {"gen": 1.0}},
            )

    trainer.async_rollout_manager = RolloutManager()

    class Client:
        async def start_grouped(self, request_id, *, prompt_ids, sampling_params, routing_key):
            class Tracked:
                backend_request_id = request_id
                server_id = "server"

                async def result(self):
                    assert awake["value"]
                    assert sampling_params["n"] == 1
                    assert sampling_params["max_tokens"] == 4
                    return [
                        SimpleNamespace(
                            token_ids=[99],
                            stop_reason="completed",
                            extra_fields={
                                "branch_id": 0,
                                "global_steps": 7,
                                "finish_reason": "stop",
                            },
                        )
                    ]

                async def abort(self):
                    return None

                async def drain(self):
                    return None

                async def release(self):
                    return None

            return Tracked()

    trainer.llm_server_manager = SimpleNamespace(get_client=lambda: Client())
    short_patterns = ([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])
    reward_call = {"short": 0}

    def score(self, batch):
        if batch.meta_info.get("boundary_reward_only", False):
            assert awake["value"]
            events.append("long_reward")
            if failure_stage == "long_reward":
                raise RuntimeError("long reward failed")
            pattern = short_patterns[(reward_call["short"] - 1) % len(short_patterns)]
            return SimpleNamespace(
                reward_tensor=torch.full((len(batch), 8), 1.0e20),
                extra_info={"acc": pattern, "score": pattern},
            )
        events.append("short_reward")
        pattern = short_patterns[reward_call["short"] % len(short_patterns)]
        reward_call["short"] += 1
        shaped = torch.zeros((len(batch), 4), dtype=torch.float32)
        shaped[:, -1] = torch.tensor(pattern)
        return SimpleNamespace(reward_tensor=shaped, extra_info={"acc": pattern, "score": pattern})

    trainer._score_batch_with_existing_reward_pipeline = MethodType(score, trainer)
    awake = {"value": False}

    class Checkpoint:
        def update_weights(self, version):
            events.append(f"publish({version})")
            awake["value"] = True

        def sleep_replicas(self):
            events.append("sleep")
            assert awake["value"]
            awake["value"] = False

    trainer.checkpoint_manager = Checkpoint()
    trainer._compute_old_and_reference = MethodType(
        lambda self, batch, _metrics, _timing: events.append("old/ref") or batch,
        trainer,
    )
    actor_batches = []

    def advantage_actor(self, batch, _metrics, _timing):
        events.extend(["GRPO", "actor_update"])
        batch.batch["advantages"] = batch.batch["token_level_rewards"].clone()
        batch.batch["returns"] = batch.batch["token_level_rewards"].clone()
        batch.batch["loss_mask"] = batch.batch["response_mask"].clone()
        actor_batches.append(batch)
        return batch, SimpleNamespace(meta_info={"metrics": {}})

    trainer._compute_advantage_and_actor_update = MethodType(advantage_actor, trainer)

    py_before = random.getstate()
    np_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    error = None
    try:
        trainer.fit()
    except RuntimeError as exc:
        error = exc
    rng_after = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.random.get_rng_state().clone())
    return SimpleNamespace(
        trainer=trainer,
        events=events,
        filter_spy=filter_spy,
        logger=logger,
        actor_batch=actor_batches[0] if actor_batches else None,
        rng_before=(py_before, np_before, torch_before),
        rng_after=rng_after,
        error=error,
    )


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def _assert_dataproto_equal(left, right):
    assert list(left.batch.keys()) == list(right.batch.keys())
    for key in left.batch.keys():
        assert torch.equal(left.batch[key], right.batch[key]), key
    assert set(left.non_tensor_batch) == set(right.non_tensor_batch)
    for key in left.non_tensor_batch:
        assert left.non_tensor_batch[key].tolist() == right.non_tensor_batch[key].tolist(), key


def test_two_batch_fit_call_order_keeps_replicas_awake_and_one_policy_version(monkeypatch):
    result = _run_fit_harness(monkeypatch, mode="replace", generation_batches=2)
    assert result.error is None
    assert result.events[:15] == [
        "publish(7)",
        "normal(7)",
        "short_reward",
        "continuation(7)",
        "long_reward",
        "filter",
        "normal(7)",
        "short_reward",
        "continuation(7)",
        "long_reward",
        "filter",
        "sleep",
        "old/ref",
        "GRPO",
        "actor_update",
    ]
    assert result.events[15] == "publish(8)"
    assert result.trainer._rollout_policy_version == 8
    assert [metric for metric, _keys, _uids in result.filter_spy] == ["boundary_acc", "boundary_acc"]
    assert result.logger.records[-1][0]["train/num_gen_batches"] == 2
    assert result.logger.records[-1][0]["boundary_return/extra_generated_tokens"] == 8.0
    assert result.actor_batch.batch["responses"].shape[-1] == 4
    assert not bool((result.actor_batch.batch["responses"] == 99).any().item())
    for label in (
        "boundary_hit_cap",
        "boundary_eligible",
        "boundary_applied",
        "boundary_changed",
        "boundary_recovered",
        "boundary_regressed",
        "boundary_task_delta",
        "boundary_group_unlocked",
    ):
        assert label in result.actor_batch.batch
    assert "boundary_group_newly_locked" not in result.actor_batch.batch


def test_single_batch_fit_call_order(monkeypatch):
    result = _run_fit_harness(monkeypatch, mode="replace", generation_batches=1)
    assert result.error is None
    assert result.events[:10] == [
        "publish(7)",
        "normal(7)",
        "short_reward",
        "continuation(7)",
        "long_reward",
        "filter",
        "sleep",
        "old/ref",
        "GRPO",
        "actor_update",
    ]
    assert result.events[10] == "publish(8)"


def test_dynamic_sampling_gate_runs_three_cycles_and_stops_before_old_ref_advantage_actor(monkeypatch, tmp_path):
    receipt = tmp_path / "gate.jsonl"
    result = _run_fit_harness(
        monkeypatch,
        mode="replace",
        generation_batches=1,
        gate_cycles=3,
        gate_receipt_path=str(receipt),
    )
    assert result.error is None
    assert "old/ref" not in result.events
    assert "GRPO" not in result.events
    assert "actor_update" not in result.events
    assert result.events.count("normal(7)") == 3
    assert result.events.count("sleep") == 3
    assert result.events.count("publish(7)") == 3
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["cycle"] for record in records] == [1, 2, 3]
    assert all(record["status"] == "PASS" for record in records)


@pytest.mark.parametrize("failure_stage", ["continuation", "long_reward"])
def test_fit_failure_sleeps_once_and_never_filters_updates_or_publishes_next(monkeypatch, failure_stage):
    result = _run_fit_harness(
        monkeypatch,
        mode="replace",
        generation_batches=1,
        failure_stage=failure_stage,
    )
    assert result.error is not None
    assert result.events.count("sleep") == 1
    assert "filter" not in result.events
    assert "actor_update" not in result.events
    assert "publish(8)" not in result.events


def test_shadow_fit_is_actor_batch_filter_selection_and_rng_equivalent_to_off(monkeypatch):
    initial_py = random.getstate()
    initial_np = copy.deepcopy(np.random.get_state())
    initial_torch = torch.random.get_rng_state().clone()
    baseline = _run_fit_harness(monkeypatch, mode="off", generation_batches=2)
    random.setstate(initial_py)
    np.random.set_state(initial_np)
    torch.random.set_rng_state(initial_torch)
    shadow = _run_fit_harness(monkeypatch, mode="shadow", generation_batches=2)
    assert not any(str(key).startswith("boundary_") for key in baseline.actor_batch.batch.keys())

    assert baseline.error is shadow.error is None
    _assert_dataproto_equal(baseline.actor_batch, shadow.actor_batch)
    assert [uids for _metric, _keys, uids in baseline.filter_spy] == [
        uids for _metric, _keys, uids in shadow.filter_spy
    ]
    assert [metric for metric, _keys, _uids in shadow.filter_spy] == ["acc", "acc"]
    assert all("boundary_acc" not in keys and "boundary_task_score" not in keys for _m, keys, _u in shadow.filter_spy)
    assert (
        baseline.logger.records[-1][0]["train/num_gen_batches"] == shadow.logger.records[-1][0]["train/num_gen_batches"]
    )
    assert (
        baseline.logger.records[-1][0]["train/retained_prompt_groups"]
        == shadow.logger.records[-1][0]["train/retained_prompt_groups"]
    )
    assert shadow.logger.records[-1][0]["boundary_return/shadow_candidate_noop_gate_pass_count"] == 2.0
    _assert_rng_equal(baseline.rng_before, baseline.rng_after)
    _assert_rng_equal(shadow.rng_before, shadow.rng_after)
