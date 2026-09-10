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

from verl.workers.config.rollout import BoundaryReturnConfig

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
