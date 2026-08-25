"""Synchronous DAPO integration for natural-continuation boundary returns."""

from __future__ import annotations

import contextlib
import random
from typing import Any, Iterator

import numpy as np
import torch

from verl import DataProto
from verl.experimental.natural_continuation_boundary_return.accumulator import (
    BoundaryReturnStepAccumulator,
)
from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BoundaryRewardOutput,
    apply_boundary_return,
    build_long_reward_batch,
)
from verl.experimental.natural_continuation_boundary_return.runtime import (
    run_boundary_continuations,
)
from verl.experimental.probe_credit.dapo_trainer import (
    RayDAPOProbeCreditTrainer,
    _config_get,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.config.rollout import BoundaryReturnConfig


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
        boundary = self._boundary_config()
        boundary.validate()
        if boundary.mode == "off":
            return
        rollout = self.config.actor_rollout_ref.rollout
        algorithm = self.config.algorithm
        if _config_get(algorithm, "adv_estimator") not in (
            "grpo",
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_VECTORIZED,
        ):
            raise ValueError("boundary_return supports synchronous GRPO only")
        if _config_get(rollout, "name") != "vllm":
            raise ValueError("boundary_return supports the vLLM rollout backend only")
        if getattr(self, "use_critic", False):
            raise ValueError("boundary_return requires GRPO with no critic")
        if bool(_config_get(rollout, "ignore_eos", False)):
            raise ValueError("boundary_return requires ignore_eos=false")
        if bool(_config_get(_config_get(rollout, "multi_turn"), "enable", False)):
            raise ValueError("boundary_return supports single-turn rollout only")
        if bool(_config_get(algorithm, "use_kl_in_reward", False)):
            raise ValueError("boundary_return requires algorithm.use_kl_in_reward=false")

        short_length = int(_config_get(rollout, "response_length"))
        if boundary.long_response_length <= short_length:
            raise ValueError("boundary_return requires L > H (long_response_length > response_length)")
        max_model_len = _config_get(rollout, "max_model_len")
        prompt_length = int(_config_get(rollout, "prompt_length"))
        if max_model_len is None or prompt_length + boundary.long_response_length > int(max_model_len):
            raise ValueError(
                "boundary_return context requires max_model_len >= prompt_length + long_response_length"
            )

        filter_groups = _config_get(algorithm, "filter_groups")
        if boundary.mode == "replace":
            if not bool(_config_get(filter_groups, "enable", False)):
                raise ValueError("boundary_return replace requires filter_groups.enable=true")
            if _config_get(filter_groups, "metric") != boundary.correctness_key:
                raise ValueError(
                    "boundary_return replace requires Hydra filter_groups.metric == correctness_key"
                )

        forced_answer = _config_get(rollout, "forced_answer_probe")
        forced_credit = _config_get(forced_answer, "training_credit")
        if bool(_config_get(forced_answer, "enable", False)) or bool(
            _config_get(forced_credit, "enable", False)
        ):
            raise ValueError("boundary_return cannot be combined with forced-answer or FA-TR")
        if bool(_config_get(_config_get(algorithm, "censor_aware_advantage"), "enable", False)):
            raise ValueError("boundary_return cannot be combined with FA-CAC/FA-RAR")
        if bool(_config_get(_config_get(algorithm, "probe_credit"), "enable", False)):
            raise ValueError("boundary_return cannot be combined with Probe Credit")
        if _config_get(_config_get(algorithm, "readiness_dominance"), "mode", "off") != "off":
            raise ValueError("boundary_return cannot be combined with Readiness")
        if _config_get(_config_get(algorithm, "success_support_floor"), "mode", "off") != "off":
            raise ValueError("boundary_return cannot be combined with BSSF")
        if _config_get(
            _config_get(algorithm, "on_policy_budgeted_capability_floor"), "mode", "off"
        ) != "off":
            raise ValueError("boundary_return cannot be combined with OBCF")

    def _effective_filter_metric(self) -> str | None:
        boundary = self._boundary_config()
        if boundary.mode == "replace":
            return "boundary_acc"
        return super()._effective_filter_metric()

    def _step_accumulator(self) -> BoundaryReturnStepAccumulator:
        accumulator = getattr(self, "_boundary_return_step_accumulator", None)
        if accumulator is None:
            accumulator = BoundaryReturnStepAccumulator(
                correctness_threshold=self._boundary_config().correctness_threshold
            )
            self._boundary_return_step_accumulator = accumulator
        return accumulator

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
        rollout = self.config.actor_rollout_ref.rollout
        policy_version = self._validate_rollout_policy_version(candidate)
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
                        sampling_params={
                            "temperature": float(_config_get(rollout, "temperature")),
                            "top_p": float(_config_get(rollout, "top_p")),
                            "top_k": int(_config_get(rollout, "top_k")),
                            "repetition_penalty": float(
                                _config_get(rollout, "repetition_penalty", 1.0)
                            ),
                        },
                    )
                if capture is None:
                    raise AssertionError("active boundary_return returned no continuation capture")
                if capture.generations:
                    long_batch = build_long_reward_batch(
                        candidate,
                        capture.generations,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    long_batch.meta_info["boundary_reward_only"] = True
                    with marked_timer("boundary_long_reward", timing_raw, color="magenta"):
                        normalized = self._score_batch_with_existing_reward_pipeline(long_batch)
                    long_reward = BoundaryRewardOutput(
                        reward_tensor=normalized.reward_tensor,
                        extra_info=normalized.extra_info,
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
                self._step_accumulator().add(result)
            return candidate
        except BaseException:
            if not getattr(self, "_boundary_failure_replicas_slept", False):
                self.checkpoint_manager.sleep_replicas()
                self._boundary_failure_replicas_slept = True
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
