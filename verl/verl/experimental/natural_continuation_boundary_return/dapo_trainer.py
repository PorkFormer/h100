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

"""Synchronous DAPO integration for natural-continuation boundary returns."""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import random
from typing import Any, Iterator

import numpy as np
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import build_rollout_sampling_params
from verl.experimental.natural_continuation_boundary_return.accumulator import (
    BoundaryReturnStepAccumulator,
)
from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BoundaryRewardOutput,
    apply_boundary_return,
    build_long_reward_batch,
    extract_required_reward_scalars,
)
from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations
from verl.experimental.probe_credit.dapo_trainer import (
    RayDAPOProbeCreditTrainer,
    _config_get,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.config.rollout import BoundaryReturnConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextlib.contextmanager
def _preserve_driver_rng_state() -> Iterator[None]:
    """Isolate auxiliary inference/scoring from Python, NumPy, and Torch CPU RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def validate_boundary_return_preflight(config: Any, *, use_critic: bool | None = None) -> None:
    """Pure, fail-closed validation safe to run before any data or worker setup."""
    rollout = _config_get(_config_get(config, "actor_rollout_ref"), "rollout")
    raw_boundary = _config_get(rollout, "boundary_return", {})
    boundary = (
        raw_boundary
        if isinstance(raw_boundary, BoundaryReturnConfig)
        else omega_conf_to_dataclass(raw_boundary, BoundaryReturnConfig)
    )
    boundary.validate()
    if boundary.mode == "off":
        return

    algorithm = _config_get(config, "algorithm")
    if _config_get(algorithm, "adv_estimator") not in (
        "grpo",
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_VECTORIZED,
    ):
        raise ValueError("boundary_return supports synchronous GRPO only")
    if _config_get(rollout, "name") != "vllm":
        raise ValueError("boundary_return supports the vLLM rollout backend only")
    if _config_get(rollout, "mode") != "async":
        raise ValueError("boundary_return requires rollout.mode=async")
    if use_critic is None:
        use_critic = bool(_config_get(_config_get(config, "critic"), "enable", False))
    if use_critic:
        raise ValueError("boundary_return requires GRPO with no critic")
    if bool(_config_get(rollout, "ignore_eos", False)):
        raise ValueError("boundary_return requires ignore_eos=false")
    if bool(_config_get(_config_get(rollout, "multi_turn"), "enable", False)):
        raise ValueError("boundary_return supports single-turn rollout only")
    if bool(_config_get(algorithm, "use_kl_in_reward", False)):
        raise ValueError("boundary_return requires algorithm.use_kl_in_reward=false")

    agent = _config_get(rollout, "agent")
    if _config_get(agent, "default_agent_loop") != "single_turn_agent":
        raise ValueError("boundary_return requires agent.default_agent_loop=single_turn_agent")
    if _config_get(agent, "agent_loop_config_path") is not None or _config_get(
        agent, "agent_loop_manager_class"
    ) is not None:
        raise ValueError("boundary_return does not support a custom agent loop")
    custom_server = _config_get(agent, "custom_async_server")
    if _config_get(custom_server, "path") is not None or _config_get(custom_server, "name") is not None:
        raise ValueError("boundary_return does not support a custom async server")

    reward = _config_get(config, "reward")
    reward_manager = _config_get(reward, "reward_manager")
    if _config_get(reward_manager, "source") != "register":
        raise ValueError("boundary_return requires the registered DAPO reward manager")
    if _config_get(reward_manager, "name") != "dapo":
        raise ValueError("boundary_return requires the DAPO reward manager")
    if boundary.correctness_key != "acc":
        raise ValueError("boundary_return v1 requires correctness_key=acc")
    if boundary.task_score_key != "score":
        raise ValueError("boundary_return v1 requires task_score_key=score")
    if _config_get(_config_get(reward, "custom_reward_function"), "path") is not None:
        raise ValueError("boundary_return v1 does not support a custom reward function path")
    if bool(_config_get(_config_get(reward, "reward_model"), "enable", False)):
        raise ValueError("boundary_return v1 does not support the reward model path")
    if _config_get(_config_get(reward, "sandbox_fusion"), "url") is not None:
        raise ValueError("boundary_return v1 does not support the sandbox reward path")
    if bool(_config_get(_config_get(config, "distillation"), "enabled", False)):
        raise ValueError("boundary_return v1 does not support distillation or a teacher policy")
    rollout_correction = _config_get(algorithm, "rollout_correction")
    if rollout_correction is not None and any(
        (
            _config_get(rollout_correction, "rollout_is") is not None,
            _config_get(rollout_correction, "rollout_rs") is not None,
            bool(_config_get(rollout_correction, "bypass_mode", False)),
        )
    ):
        raise ValueError("boundary_return v1 does not support rollout correction")
    if _config_get(_config_get(config, "global_profiler"), "steps"):
        raise ValueError("boundary_return v1 does not support configured profiling steps")

    short_length = int(_config_get(rollout, "response_length"))
    if boundary.long_response_length <= short_length:
        raise ValueError("boundary_return requires L > H (long_response_length > response_length)")
    max_model_len = _config_get(rollout, "max_model_len")
    prompt_length = int(_config_get(rollout, "prompt_length"))
    if max_model_len is None or prompt_length + boundary.long_response_length > int(max_model_len):
        raise ValueError("boundary_return context requires max_model_len >= prompt_length + long_response_length")

    filter_groups = _config_get(algorithm, "filter_groups")
    if boundary.mode == "replace":
        if not bool(_config_get(filter_groups, "enable", False)):
            raise ValueError("boundary_return replace requires filter_groups.enable=true")
        if _config_get(filter_groups, "metric") != boundary.correctness_key:
            raise ValueError("boundary_return replace requires Hydra filter_groups.metric == correctness_key")

    forced_answer = _config_get(rollout, "forced_answer_probe")
    forced_credit = _config_get(forced_answer, "training_credit")
    if bool(_config_get(forced_answer, "enable", False)) or bool(_config_get(forced_credit, "enable", False)):
        raise ValueError("boundary_return cannot be combined with forced-answer or FA-TR")
    if bool(_config_get(_config_get(algorithm, "censor_aware_advantage"), "enable", False)):
        raise ValueError("boundary_return cannot be combined with FA-CAC/FA-RAR")
    if bool(_config_get(_config_get(algorithm, "probe_credit"), "enable", False)):
        raise ValueError("boundary_return cannot be combined with Probe Credit")
    if _config_get(_config_get(algorithm, "readiness_dominance"), "mode", "off") != "off":
        raise ValueError("boundary_return cannot be combined with Readiness")
    if _config_get(_config_get(algorithm, "success_support_floor"), "mode", "off") != "off":
        raise ValueError("boundary_return cannot be combined with BSSF")
    if _config_get(_config_get(algorithm, "on_policy_budgeted_capability_floor"), "mode", "off") != "off":
        raise ValueError("boundary_return cannot be combined with OBCF")


def _contains_multimodal_payload(candidate: DataProto) -> bool:
    def nonempty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, np.ndarray):
            return any(nonempty(item) for item in value.reshape(-1).tolist())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        if torch.is_tensor(value):
            return value.numel() > 0
        return True

    return any(
        key in candidate.non_tensor_batch and nonempty(candidate.non_tensor_batch[key])
        for key in ("multi_modal_data", "multi_modal_inputs", "images", "videos", "audios")
    )


def _assert_shadow_candidate_noop(before: DataProto, after: DataProto) -> None:
    def equal(left: Any, right: Any) -> bool:
        if torch.is_tensor(left) and torch.is_tensor(right):
            return bool(torch.equal(left, right))
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return left.dtype == right.dtype and left.shape == right.shape and left.tolist() == right.tolist()
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
        return bool(left == right)

    if list(before.batch.keys()) != list(after.batch.keys()):
        raise AssertionError("shadow candidate tensor keys changed")
    for key in before.batch.keys():
        if not torch.equal(before.batch[key], after.batch[key]):
            raise AssertionError(f"shadow candidate tensor {key!r} changed")
    if set(before.non_tensor_batch) != set(after.non_tensor_batch):
        raise AssertionError("shadow candidate non-tensor keys changed")
    for key, value in before.non_tensor_batch.items():
        if np.asarray(value, dtype=object).tolist() != np.asarray(
            after.non_tensor_batch[key], dtype=object
        ).tolist():
            raise AssertionError(f"shadow candidate non-tensor field {key!r} changed")
    if not equal(before.meta_info, after.meta_info):
        raise AssertionError("shadow candidate meta_info changed")


class RayDAPOBoundaryReturnTrainer(RayDAPOProbeCreditTrainer):
    """Override only the two DAPO candidate/filter hooks; inherit the full fit loop."""

    def _boundary_config(self) -> BoundaryReturnConfig:
        cached = getattr(self, "_typed_boundary_return_config", None)
        if cached is None:
            raw = _config_get(self.config.actor_rollout_ref.rollout, "boundary_return", {})
            cached = raw if isinstance(raw, BoundaryReturnConfig) else omega_conf_to_dataclass(
                raw, BoundaryReturnConfig
            )
            self._typed_boundary_return_config = cached
        return cached

    def _validate_probe_credit_mode(self) -> None:
        super()._validate_probe_credit_mode()
        validate_boundary_return_preflight(self.config, use_critic=getattr(self, "use_critic", False))

    def _effective_filter_metric(self) -> str | None:
        boundary = self._boundary_config()
        if boundary.mode == "replace":
            logger.info("boundary_return event=filter metric=boundary_acc")
            return "boundary_acc"
        if boundary.mode == "shadow":
            logger.info("boundary_return event=filter metric=acc mode=shadow")
        return super()._effective_filter_metric()

    def _step_accumulator(self) -> BoundaryReturnStepAccumulator:
        accumulator = getattr(self, "_boundary_return_step_accumulator", None)
        if accumulator is None:
            accumulator = BoundaryReturnStepAccumulator(
                correctness_threshold=self._boundary_config().correctness_threshold
            )
            self._boundary_return_step_accumulator = accumulator
        return accumulator

    def _score_long_generations_in_chunks(
        self,
        candidate: DataProto,
        generations: tuple[Any, ...],
        boundary: BoundaryReturnConfig,
    ) -> BoundaryRewardOutput:
        correctness_parts: list[np.ndarray] = []
        task_score_parts: list[np.ndarray] = []
        for start in range(0, len(generations), boundary.long_reward_chunk_size):
            chunk = generations[start : start + boundary.long_reward_chunk_size]
            long_batch = build_long_reward_batch(
                candidate,
                chunk,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            long_batch.meta_info["boundary_reward_only"] = True
            normalized = self._score_batch_with_existing_reward_pipeline(long_batch)
            scalars = extract_required_reward_scalars(
                BoundaryRewardOutput(
                    reward_tensor=normalized.reward_tensor,
                    extra_info=normalized.extra_info,
                ),
                expected_count=len(chunk),
                correctness_key=boundary.correctness_key,
                task_score_key=boundary.task_score_key,
            )
            correctness_parts.append(scalars.correctness)
            task_score_parts.append(scalars.task_score)
        return BoundaryRewardOutput(
            reward_tensor=torch.empty((len(generations), 0), dtype=torch.float32),
            extra_info={
                boundary.correctness_key: np.concatenate(correctness_parts),
                boundary.task_score_key: np.concatenate(task_score_parts),
            },
        )

    def _process_candidate_after_reward_before_filter(
        self,
        candidate: DataProto,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
        generation_batch_index: int,
    ) -> DataProto:
        del metrics, generation_batch_index
        boundary = self._boundary_config()
        if boundary.mode == "off":
            return candidate
        agent_names = candidate.non_tensor_batch.get("agent_name")
        if agent_names is not None and any(
            str(agent_name) != "single_turn_agent" for agent_name in agent_names
        ):
            raise ValueError("boundary_return v1 requires every row to use single_turn_agent")
        if _contains_multimodal_payload(candidate):
            raise ValueError("boundary_return v1 does not support multimodal input rows")
        shadow_snapshot = (
            copy.deepcopy(candidate)
            if boundary.mode == "shadow" and boundary.verify_shadow_candidate_noop
            else None
        )
        rollout = self.config.actor_rollout_ref.rollout
        policy_version = self._validate_rollout_policy_version(candidate)
        logger.info(
            "boundary_return event=short_reward_complete policy_version=%d mode=%s",
            policy_version,
            boundary.mode,
        )
        max_model_len = int(_config_get(rollout, "max_model_len"))
        try:
            with _preserve_driver_rng_state():
                with marked_timer("boundary_continuation", timing_raw, color="magenta"):
                    capture = run_boundary_continuations(
                        config=boundary,
                        rollout_batch=candidate,
                        client=self.llm_server_manager.get_client(),
                        eos_token_id=self.tokenizer.eos_token_id,
                        short_response_length=int(_config_get(rollout, "response_length")),
                        max_model_len=max_model_len,
                        policy_version=policy_version,
                        sampling_params=build_rollout_sampling_params(rollout),
                    )
                if capture is None:
                    raise AssertionError("active boundary_return returned no continuation capture")
                if capture.generations:
                    logger.info(
                        "boundary_return event=continuation_complete policy_version=%d request_count=%d",
                        policy_version,
                        len(capture.generations),
                    )
                    with marked_timer("boundary_long_reward", timing_raw, color="magenta"):
                        long_reward = self._score_long_generations_in_chunks(
                            candidate,
                            capture.generations,
                            boundary,
                        )
                    logger.info(
                        "boundary_return event=long_reward_complete policy_version=%d row_count=%d",
                        policy_version,
                        len(capture.generations),
                    )
                else:
                    long_reward = BoundaryRewardOutput(
                        reward_tensor=torch.empty((0, 0), dtype=torch.float32),
                        extra_info={
                            boundary.correctness_key: np.asarray([], dtype=np.float64),
                            boundary.task_score_key: np.asarray([], dtype=np.float64),
                        },
                    )
                result = apply_boundary_return(
                    candidate,
                    capture=capture,
                    long_reward_output=long_reward,
                    config=boundary,
                )
                if shadow_snapshot is not None:
                    _assert_shadow_candidate_noop(shadow_snapshot, candidate)
                    result.metrics["boundary_return/shadow_candidate_noop_gate_pass_count"] = 1.0
                self._step_accumulator().add(result)
                logger.info(
                    "boundary_return event=mode_complete policy_version=%d mode=%s",
                    policy_version,
                    boundary.mode,
                )
            return candidate
        except BaseException as error:
            cleanup_attested = getattr(error, "boundary_remote_cleanup_attested", True)
            if cleanup_attested and not getattr(self, "_boundary_failure_replicas_slept", False):
                self.checkpoint_manager.sleep_replicas()
                self._boundary_failure_replicas_slept = True
            elif not cleanup_attested:
                error.add_note(
                    "rollout replicas were not slept because remote continuation drain/release "
                    "could not be attested"
                )
            raise

    def _prepare_final_retained_batch(
        self,
        batch: DataProto,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
    ) -> DataProto:
        batch = super()._prepare_final_retained_batch(batch, metrics, timing_raw)
        if self._boundary_config().mode != "off":
            step_metrics = self._step_accumulator().metrics()
            if step_metrics.get("boundary_return/prefix_penalty_drift_max") != 0.0:
                raise AssertionError("boundary_return prefix penalty drift must be exactly zero")
            metrics.update(step_metrics)
            self._boundary_return_step_accumulator = None
        return batch
