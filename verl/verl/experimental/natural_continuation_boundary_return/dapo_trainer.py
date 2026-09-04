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
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import build_rollout_sampling_params
from verl.experimental.natural_continuation_boundary_return.accumulator import (
    BoundaryReturnStepAccumulator,
)
from verl.experimental.natural_continuation_boundary_return.profiling import (
    IntervalRecorder,
    analyze_interval_dag,
    coordinate_profile_extension,
)
from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BOUNDARY_NUMERIC_TOLERANCE,
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
from verl.trainer.ppo.forced_answer_probe import detect_hit_response_cap
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.config.rollout import BoundaryReturnConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _emit_audit_event(message: str, *args: Any, level: int = logging.INFO) -> None:
    """Emit step-order evidence even when Ray worker logging is not forwarded."""
    rendered = message % args
    logger.log(level, rendered)
    print(rendered, flush=True)


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
    gate_cycles = int(_config_get(_config_get(config, "trainer"), "dynamic_sampling_gate_cycles", 0))
    if gate_cycles not in (0, 3):
        raise ValueError("Dynamic Sampling Gate 0 requires exactly 3 cycles or must be disabled with 0")
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
    if (
        _config_get(agent, "agent_loop_config_path") is not None
        or _config_get(agent, "agent_loop_manager_class") is not None
    ):
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
        if isinstance(value, dict | list | tuple | set):
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
        if isinstance(left, list | tuple) and isinstance(right, type(left)):
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
        if np.asarray(value, dtype=object).tolist() != np.asarray(after.non_tensor_batch[key], dtype=object).tolist():
            raise AssertionError(f"shadow candidate non-tensor field {key!r} changed")
    if not equal(before.meta_info, after.meta_info):
        raise AssertionError("shadow candidate meta_info changed")


class RayDAPOBoundaryReturnTrainer(RayDAPOProbeCreditTrainer):
    """Override only the two DAPO candidate/filter hooks; inherit the full fit loop."""

    def fit(self):
        panel_path = _config_get(self.config.trainer, "mechanism_panel_path")
        if panel_path not in (None, "null"):
            return self._run_mechanism_panel(Path(str(panel_path)))
        return super().fit()

    def _run_mechanism_panel(self, panel_path: Path) -> None:
        """Exercise continuation and long reward on a frozen Base cap-prefix panel."""
        boundary = self._boundary_config()
        if boundary.mode != "replace":
            raise ValueError("the mechanism panel is restricted to boundary_return.mode=replace")
        receipt_value = _config_get(self.config.trainer, "mechanism_panel_receipt_path")
        if receipt_value in (None, "null"):
            raise ValueError("the mechanism panel requires a receipt path")
        receipt_path = Path(str(receipt_value))
        row_path = receipt_path.with_suffix(".rows.jsonl")
        if receipt_path.exists() or row_path.exists():
            raise FileExistsError("refusing to overwrite frozen mechanism-panel receipts")
        rows = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < 20:
            raise ValueError("mechanism panel requires at least 20 cap-prefix requests")
        if any(
            len(row["prefix_token_ids"]) != int(self.config.actor_rollout_ref.rollout.response_length)
            for row in rows
        ):
            raise ValueError("mechanism panel contains a non-H=2048 prefix")

        prompt_width = max(len(row["prompt_token_ids"]) for row in rows)
        response_width = int(self.config.actor_rollout_ref.rollout.response_length)
        pad = int(self.tokenizer.pad_token_id)
        prompts = torch.full((len(rows), prompt_width), pad, dtype=torch.long)
        responses = torch.full((len(rows), response_width), pad, dtype=torch.long)
        prompt_mask = torch.zeros_like(prompts)
        response_mask = torch.ones_like(responses)
        for index, row in enumerate(rows):
            prompt = torch.tensor(row["prompt_token_ids"], dtype=torch.long)
            response = torch.tensor(row["prefix_token_ids"], dtype=torch.long)
            prompts[index, -len(prompt) :] = prompt
            prompt_mask[index, -len(prompt) :] = 1
            responses[index] = response
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
        input_ids = torch.cat((prompts, responses), dim=-1)
        position_ids = torch.clamp(attention_mask.cumsum(-1) - 1, min=0)
        context_keys = sorted({key for row in rows for key in row.get("source_context", {})})
        non_tensor = {
            "uid": np.asarray([str(row["prompt_id"]) for row in rows], dtype=object),
            "trajectory_id": np.asarray([str(row["trajectory_id"]) for row in rows], dtype=object),
            "finish_reason": np.asarray(["length"] * len(rows), dtype=object),
            "rollout_policy_version": np.asarray([0] * len(rows), dtype=object),
        }
        for key in context_keys:
            non_tensor[key] = np.asarray([row.get("source_context", {}).get(key) for row in rows], dtype=object)
        candidate = DataProto(
            batch=TensorDict(
                {
                    "prompts": prompts,
                    "responses": responses,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "response_mask": response_mask,
                },
                batch_size=len(rows),
            ),
            non_tensor_batch=non_tensor,
            meta_info={"temperature": float(self.config.actor_rollout_ref.rollout.temperature)},
        )
        prefix_before = candidate.batch["responses"].clone()
        self.global_steps = 0
        self._load_checkpoint()
        self._publish_rollout_policy_version(self.global_steps)
        cleanup_ack = False
        try:
            capture = run_boundary_continuations(
                config=boundary,
                rollout_batch=candidate,
                client=self.llm_server_manager.get_client(),
                eos_token_id=self.tokenizer.eos_token_id,
                short_response_length=response_width,
                max_model_len=int(self.config.actor_rollout_ref.rollout.max_model_len),
                policy_version=self.global_steps,
                sampling_params=build_rollout_sampling_params(self.config.actor_rollout_ref.rollout),
            )
            if capture is None or len(capture.generations) != len(rows):
                raise AssertionError("mechanism panel did not return exactly one continuation per frozen prefix")
            long_reward = self._score_long_generations_in_chunks(candidate, capture.generations, boundary)
            scores = extract_required_reward_scalars(
                long_reward,
                expected_count=len(rows),
                correctness_key=boundary.correctness_key,
                task_score_key=boundary.task_score_key,
            )
            if not torch.equal(prefix_before, candidate.batch["responses"]):
                raise AssertionError("mechanism panel leaked continuation tails into prefix actor tensors")
            actual_versions = {generation.actual_policy_version for generation in capture.generations}
            if actual_versions != {self.global_steps}:
                raise AssertionError(f"mechanism panel policy-version mismatch: {actual_versions}")
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with row_path.open("x", encoding="utf-8") as stream:
                for row, generation, correctness, task_score in zip(
                    rows,
                    capture.generations,
                    scores.correctness,
                    scores.task_score,
                    strict=True,
                ):
                    stream.write(
                        json.dumps(
                            {
                                "prompt_id": row["prompt_id"],
                                "trajectory_id": row["trajectory_id"],
                                "request_id": generation.request_id,
                                "policy_version": generation.actual_policy_version,
                                "prefix_tokens": len(generation.prefix_token_ids),
                                "tail_tokens": len(generation.tail_token_ids),
                                "long_correctness": float(correctness),
                                "long_task_score": float(task_score),
                                "finish_reason": generation.finish_reason,
                                "stop_reason": generation.stop_reason,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            receipt = {
                "schema_version": "qwen3-1p7b-ncbr-mechanism-panel-v1",
                "status": "PASS",
                "request_count": len(capture.requests),
                "generation_count": len(capture.generations),
                "long_reward_rows": len(scores.correctness),
                "policy_version": self.global_steps,
                "prefix_actor_tensor_unchanged": True,
                "actor_update_invoked": False,
                "tail_decode_tokens": sum(len(item.tail_token_ids) for item in capture.generations),
                "continuation_input_tokens": sum(len(item.input_token_ids) for item in capture.requests),
                "long_reward_full_response_tokens": sum(
                    len(item.prefix_token_ids) + len(item.tail_token_ids) for item in capture.generations
                ),
                "profiling_intervals": [
                    interval.to_dict() for interval in (*capture.profiling_intervals, *long_reward.profiling_intervals)
                ],
                "rows": str(row_path.resolve()),
            }
        finally:
            self.checkpoint_manager.sleep_replicas()
            cleanup_ack = True
        receipt["replica_sleep_cleanup_ack"] = cleanup_ack
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _boundary_config(self) -> BoundaryReturnConfig:
        cached = getattr(self, "_typed_boundary_return_config", None)
        if cached is None:
            raw = _config_get(self.config.actor_rollout_ref.rollout, "boundary_return", {})
            cached = (
                raw if isinstance(raw, BoundaryReturnConfig) else omega_conf_to_dataclass(raw, BoundaryReturnConfig)
            )
            self._typed_boundary_return_config = cached
        return cached

    def _validate_probe_credit_mode(self) -> None:
        super()._validate_probe_credit_mode()
        validate_boundary_return_preflight(self.config, use_critic=getattr(self, "use_critic", False))

    def _effective_filter_metric(self) -> str | None:
        boundary = self._boundary_config()
        if boundary.mode == "replace":
            _emit_audit_event("boundary_return event=filter metric=boundary_acc")
            return "boundary_acc"
        if boundary.mode == "shadow":
            _emit_audit_event("boundary_return event=filter metric=acc mode=shadow")
        return super()._effective_filter_metric()

    def _publish_rollout_policy_version(self, version: int) -> None:
        _emit_audit_event(
            "boundary_return event=publish_start policy_version=%d mode=%s",
            int(version),
            self._boundary_config().mode,
        )
        super()._publish_rollout_policy_version(version)
        _emit_audit_event(
            "boundary_return event=publish_complete policy_version=%d mode=%s",
            int(version),
            self._boundary_config().mode,
        )

    def _step_accumulator(self) -> BoundaryReturnStepAccumulator:
        accumulator = getattr(self, "_boundary_return_step_accumulator", None)
        if accumulator is None:
            accumulator = BoundaryReturnStepAccumulator(
                correctness_threshold=self._boundary_config().correctness_threshold
            )
            self._boundary_return_step_accumulator = accumulator
        return accumulator

    @contextlib.contextmanager
    def _profile_candidate_stage(self, name: str, metadata: dict[str, Any]) -> Iterator[dict[str, Any]]:
        recorder = IntervalRecorder(f"boundary-{name}-step-{self.global_steps}")
        with recorder.record(name, metadata=metadata):
            yield metadata
        profile_intervals = getattr(self, "_boundary_profile_intervals", None)
        if profile_intervals is None:
            profile_intervals = []
            self._boundary_profile_intervals = profile_intervals
        profile_intervals.extend(recorder.intervals)

    def _profile_actor_metadata(self, batch: DataProto) -> dict[str, Any]:
        return {"actor_valid_tokens": int(batch.batch["response_mask"].sum().item())}

    def _after_normal_rollout(self, candidate: DataProto) -> None:
        """Persist Base H=2048 cap rows without changing the candidate batch."""
        path_value = _config_get(_config_get(self.config, "trainer", {}), "hard_prefix_source_path")
        if path_value in (None, "null"):
            return
        if self._boundary_config().mode != "off":
            raise ValueError("hard-prefix source collection is restricted to the Baseline arm")
        path = Path(str(path_value))
        path.parent.mkdir(parents=True, exist_ok=True)
        initialized = bool(getattr(self, "_hard_prefix_source_initialized", False))
        if not initialized and path.exists():
            raise FileExistsError(f"refusing to overwrite hard-prefix source: {path}")

        prompt_width = candidate.batch["prompts"].shape[-1]
        prompt_mask = candidate.batch["attention_mask"][:, :prompt_width].bool()
        response_mask = candidate.batch["response_mask"].bool()
        finish_reasons = candidate.non_tensor_batch.get("finish_reason")
        if finish_reasons is not None and len(finish_reasons) != len(candidate):
            raise ValueError("hard-prefix source finish_reason must be row-aligned when present")
        trajectory_ids = candidate.non_tensor_batch.get("trajectory_id")
        if trajectory_ids is None or len(trajectory_ids) != len(candidate):
            raise ValueError("hard-prefix source requires row-aligned trajectory_id")

        def safe(value: Any) -> Any:
            if isinstance(value, np.generic):
                return safe(value.item())
            if isinstance(value, np.ndarray):
                return [safe(item) for item in value.tolist()]
            if torch.is_tensor(value):
                return safe(value.detach().cpu().tolist())
            if isinstance(value, dict):
                return {str(key): safe(item) for key, item in value.items()}
            if isinstance(value, list | tuple):
                return [safe(item) for item in value]
            if value is None or isinstance(value, str | int | float | bool):
                return value
            return repr(value)

        rows = []
        expected_length = int(self.config.actor_rollout_ref.rollout.response_length)
        policy_version = self._validate_rollout_policy_version(candidate)
        response_tokens = [
            candidate.batch["responses"][row][response_mask[row]].detach().cpu().tolist()
            for row in range(len(candidate))
        ]
        hit_cap = detect_hit_response_cap(
            finish_reasons=finish_reasons,
            response_lengths=[len(tokens) for tokens in response_tokens],
            max_response_length=expected_length,
            response_token_ids=response_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        for row in range(len(candidate)):
            response = response_tokens[row]
            if not hit_cap[row]:
                continue
            prompt = candidate.batch["prompts"][row][prompt_mask[row]].detach().cpu().tolist()
            prompt_digest = hashlib.sha256(
                ",".join(str(int(token)) for token in prompt).encode("ascii")
            ).hexdigest()
            prompt_id: Any = prompt_digest
            for key in ("prompt_id", "dataset_row_id", "index"):
                values = candidate.non_tensor_batch.get(key)
                if values is not None and len(values) == len(candidate):
                    prompt_id = safe(values[row])
                    break
            source_context = {
                key: safe(values[row])
                for key in ("data_source", "reward_model", "extra_info", "raw_prompt")
                if (values := candidate.non_tensor_batch.get(key)) is not None and len(values) == len(candidate)
            }
            rows.append(
                {
                    "prompt_id": str(prompt_id),
                    "prompt_token_ids": [int(token) for token in prompt],
                    "response_token_ids": [int(token) for token in response],
                    "trajectory_id": str(trajectory_ids[row]),
                    "finish_reason": "length",
                    "raw_finish_reason": safe(finish_reasons[row]) if finish_reasons is not None else None,
                    "cap_detection": "backend_reason_or_full_without_eos",
                    "policy_version": policy_version,
                    "source_context": source_context,
                }
            )
        with path.open("a" if initialized else "x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._hard_prefix_source_initialized = True

    def _write_candidate_mechanism_rows(
        self,
        candidate: DataProto,
        result: Any,
        generation_batch_index: int,
    ) -> None:
        path_value = _config_get(_config_get(self.config, "trainer", {}), "mechanism_rows_path")
        if path_value in (None, "null"):
            return
        path = Path(str(path_value))
        path.parent.mkdir(parents=True, exist_ok=True)
        initialized = bool(getattr(self, "_mechanism_rows_initialized", False))
        if not initialized and path.exists():
            raise FileExistsError(f"refusing to overwrite candidate mechanism rows: {path}")
        grouped_short: dict[str, list[float]] = {}
        grouped_boundary: dict[str, list[float]] = {}
        for uid, short, boundary in zip(
            result.uids.tolist(),
            result.short_acc.tolist(),
            result.boundary_acc.tolist(),
            strict=True,
        ):
            grouped_short.setdefault(str(uid), []).append(float(short))
            grouped_boundary.setdefault(str(uid), []).append(float(boundary))
        newly_locked = {
            uid
            for uid, values in grouped_short.items()
            if np.std(np.asarray(values, dtype=np.float64)) > BOUNDARY_NUMERIC_TOLERANCE
            and np.std(np.asarray(grouped_boundary[uid], dtype=np.float64)) <= BOUNDARY_NUMERIC_TOLERANCE
        }
        trajectory_ids = candidate.non_tensor_batch["trajectory_id"]
        threshold = self._boundary_config().correctness_threshold
        with path.open("a" if initialized else "x", encoding="utf-8") as stream:
            for row in np.flatnonzero(result.hit_response_cap).tolist():
                short_success = result.short_acc[row] >= threshold
                long_success = result.long_acc[row] >= threshold
                transition = (
                    f"h_{'correct' if short_success else 'wrong'}_"
                    f"l_{'correct' if long_success else 'wrong'}"
                )
                uid = str(result.uids[row])
                stream.write(
                    json.dumps(
                        {
                            "global_step": int(self.global_steps),
                            "generation_batch_index": int(generation_batch_index),
                            "uid": uid,
                            "trajectory_id": str(trajectory_ids[row]),
                            "transition_class": transition,
                            "tail_length": int(result.tail_token_lengths[row]),
                            "short_acc": float(result.short_acc[row]),
                            "long_acc": float(result.long_acc[row]),
                            "boundary_acc": float(result.boundary_acc[row]),
                            "task_delta": float(result.task_score_delta[row]),
                            "long_hit_cap": bool(result.long_hit_response_cap[row]),
                            "boundary_group_newly_locked": uid in newly_locked,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        self._mechanism_rows_initialized = True

    def _score_long_generations_in_chunks(
        self,
        candidate: DataProto,
        generations: tuple[Any, ...],
        boundary: BoundaryReturnConfig,
    ) -> BoundaryRewardOutput:
        correctness_parts: list[np.ndarray] = []
        task_score_parts: list[np.ndarray] = []
        reward_loop_manager = getattr(self, "reward_loop_manager", None)
        reward_loop_workers = getattr(reward_loop_manager, "reward_loop_workers", ())
        reward_worker_count = len(reward_loop_workers) or 1
        global_step = int(getattr(self, "global_steps", 0))
        recorder = IntervalRecorder(f"boundary-long-reward-step-{global_step}")
        with recorder.record(
            "boundary_long_reward",
            metadata={"rows": len(generations), "chunk_size": boundary.long_reward_chunk_size},
        ):
            for chunk_index, start in enumerate(range(0, len(generations), boundary.long_reward_chunk_size)):
                chunk = generations[start : start + boundary.long_reward_chunk_size]
                full_tokens = sum(
                    len(getattr(item, "prefix_token_ids", ())) + len(getattr(item, "tail_token_ids", ()))
                    for item in chunk
                )
                with recorder.record(
                    "boundary_long_reward_chunk",
                    metadata={"chunk_index": chunk_index, "rows": len(chunk), "full_response_tokens": full_tokens},
                ):
                    with recorder.record(
                        "long_reward_batch_build",
                        metadata={"chunk_index": chunk_index, "rows": len(chunk), "full_response_tokens": full_tokens},
                    ):
                        long_batch = build_long_reward_batch(
                            candidate,
                            chunk,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        long_batch.meta_info["boundary_reward_only"] = True
                        original_count = len(long_batch)
                        padding_count = (-original_count) % reward_worker_count
                        long_batch.padding(padding_count, padding_candidate="last")
                    with recorder.record(
                        "long_reward_model_forward",
                        metadata={"chunk_index": chunk_index, "rows": original_count, "padded_rows": len(long_batch)},
                    ):
                        normalized = self._score_batch_with_existing_reward_pipeline(long_batch)
                    scalars = extract_required_reward_scalars(
                        BoundaryRewardOutput(
                            reward_tensor=normalized.reward_tensor,
                            extra_info=normalized.extra_info,
                        ),
                        expected_count=len(long_batch),
                        correctness_key=boundary.correctness_key,
                        task_score_key=boundary.task_score_key,
                    )
                    correctness_parts.append(scalars.correctness[:original_count])
                    task_score_parts.append(scalars.task_score[:original_count])
        return BoundaryRewardOutput(
            reward_tensor=torch.empty((len(generations), 0), dtype=torch.float32),
            extra_info={
                boundary.correctness_key: np.concatenate(correctness_parts),
                boundary.task_score_key: np.concatenate(task_score_parts),
            },
            profiling_intervals=tuple(recorder.intervals),
        )

    def _process_candidate_after_reward_before_filter(
        self,
        candidate: DataProto,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
        generation_batch_index: int,
    ) -> DataProto:
        del metrics
        boundary = self._boundary_config()
        if boundary.mode == "off":
            return candidate
        agent_names = candidate.non_tensor_batch.get("agent_name")
        if agent_names is not None and any(str(agent_name) != "single_turn_agent" for agent_name in agent_names):
            raise ValueError("boundary_return v1 requires every row to use single_turn_agent")
        if _contains_multimodal_payload(candidate):
            raise ValueError("boundary_return v1 does not support multimodal input rows")
        shadow_snapshot = (
            copy.deepcopy(candidate) if boundary.mode == "shadow" and boundary.verify_shadow_candidate_noop else None
        )
        rollout = self.config.actor_rollout_ref.rollout
        policy_version = self._validate_rollout_policy_version(candidate)
        _emit_audit_event(
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
                    _emit_audit_event(
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
                    _emit_audit_event(
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
                self._write_candidate_mechanism_rows(candidate, result, generation_batch_index)
                profile_intervals = getattr(self, "_boundary_profile_intervals", None)
                if profile_intervals is None:
                    profile_intervals = []
                    self._boundary_profile_intervals = profile_intervals
                profile_intervals.extend(result.profiling_intervals)
                _emit_audit_event(
                    "boundary_return event=mode_complete policy_version=%d mode=%s",
                    policy_version,
                    boundary.mode,
                )
            return candidate
        except BaseException as error:
            cleanup_attested = getattr(error, "boundary_remote_cleanup_attested", True)
            if cleanup_attested and not getattr(self, "_boundary_failure_replicas_slept", False):
                _emit_audit_event(
                    "boundary_return cleanup event=replica_sleep_start error_type=%s",
                    type(error).__name__,
                    level=logging.WARNING,
                )
                self.checkpoint_manager.sleep_replicas()
                self._boundary_failure_replicas_slept = True
                _emit_audit_event(
                    "boundary_return cleanup event=replica_sleep_ack error_type=%s",
                    type(error).__name__,
                    level=logging.WARNING,
                )
            elif not cleanup_attested:
                error.add_note(
                    "rollout replicas were not slept because remote continuation drain/release could not be attested"
                )
            raise

    def _prepare_final_retained_batch(
        self,
        batch: DataProto,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
    ) -> DataProto:
        _emit_audit_event(
            "boundary_return event=replica_sleep_start policy_version=%d mode=%s",
            self._validate_rollout_policy_version(batch),
            self._boundary_config().mode,
        )
        recorder = IntervalRecorder(f"boundary-finalize-step-{self.global_steps}")
        with recorder.record("candidate_finalization_and_replica_sleep"):
            batch = super()._prepare_final_retained_batch(batch, metrics, timing_raw)
        profile_intervals = getattr(self, "_boundary_profile_intervals", None)
        if profile_intervals is None:
            profile_intervals = []
            self._boundary_profile_intervals = profile_intervals
        profile_intervals.extend(recorder.intervals)
        _emit_audit_event(
            "boundary_return event=replica_sleep_complete policy_version=%d mode=%s",
            self._validate_rollout_policy_version(batch),
            self._boundary_config().mode,
        )
        if self._boundary_config().mode != "off":
            step_metrics = self._step_accumulator().metrics()
            if step_metrics.get("boundary_return/prefix_penalty_drift_max") != 0.0:
                raise AssertionError("boundary_return prefix penalty drift must be exactly zero")
            metrics.update(step_metrics)
            self._boundary_return_step_accumulator = None
            if self._boundary_config().mode == "replace":
                required_labels = {
                    "boundary_hit_cap",
                    "boundary_eligible",
                    "boundary_applied",
                    "boundary_changed",
                    "boundary_recovered",
                    "boundary_regressed",
                    "boundary_task_delta",
                    "boundary_group_unlocked",
                }
                missing = required_labels - set(batch.batch.keys())
                if missing:
                    raise AssertionError(f"retained boundary actor batch is missing row labels: {sorted(missing)}")
                if (
                    "boundary_group_newly_locked" in batch.batch
                    or "boundary_group_newly_locked" in batch.non_tensor_batch
                ):
                    raise AssertionError("newly locked groups are candidate-only diagnostics, never actor cohorts")
        else:
            forbidden = [key for key in batch.batch.keys() if str(key).startswith("boundary_")]
            if forbidden:
                raise AssertionError(f"baseline actor batch unexpectedly carries boundary labels: {forbidden}")
            metrics["boundary_return/continuation_request_count"] = 0.0
        return batch

    def _flush_profile_intervals(
        self,
        *,
        stage: str,
        timing_raw: dict[str, float] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        profile_path_value = _config_get(self.config.trainer, "profile_interval_path")
        profile_intervals = getattr(self, "_boundary_profile_intervals", [])
        if profile_path_value is not None:
            profile_path = Path(str(profile_path_value))
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": "ncbr-profile-interval-dag-v1",
                "global_step": int(self.global_steps),
                "stage": stage,
                "intervals": [interval.to_dict() for interval in profile_intervals],
                "analysis": analyze_interval_dag(profile_intervals),
                "trainer_timing_raw": dict(timing_raw or {}),
                "step_metrics": dict(metrics or {}),
            }
            with profile_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._boundary_profile_intervals = []

    def _write_dynamic_sampling_gate_receipt(self, record: dict[str, Any]) -> None:
        path_value = _config_get(self.config.trainer, "dynamic_sampling_gate_receipt_path")
        if path_value is None:
            raise ValueError("Dynamic Sampling Gate 0 requires trainer.dynamic_sampling_gate_receipt_path")
        path = Path(str(path_value))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **record,
            "arm": "baseline" if self._boundary_config().mode == "off" else "v1",
            "boundary_return_mode": self._boundary_config().mode,
            "global_step": int(self.global_steps),
            "pid": os.getpid(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._flush_profile_intervals(stage="dynamic_sampling_gate")

    def _flush_step_profile(self, timing_raw: dict[str, float], metrics: dict[str, float]) -> None:
        step_wall = timing_raw.get("step")
        if step_wall is not None:
            walls = getattr(self, "_boundary_profile_step_walls", None)
            if walls is None:
                walls = []
                self._boundary_profile_step_walls = walls
            walls.append(float(step_wall))
        self._flush_profile_intervals(stage="training", timing_raw=timing_raw, metrics=metrics)

    def _profile_should_stop_after_step(self) -> bool:
        trainer = self.config.trainer
        coordination_value = _config_get(trainer, "profile_coordination_dir")
        if coordination_value in (None, "null"):
            return False
        min_steps = int(_config_get(trainer, "profile_min_steps", 4))
        max_steps = int(_config_get(trainer, "profile_max_steps", 6))
        if (min_steps, max_steps) != (4, 6):
            raise ValueError("paired profiling is fixed to a Step 4 decision and Step 6 maximum")
        if self.global_steps != min_steps:
            return False
        walls = list(getattr(self, "_boundary_profile_step_walls", ()))
        arm = str(_config_get(trainer, "profile_arm"))
        candidate = str(_config_get(trainer, "profile_candidate"))
        diagnostics_mode = str(_config_get(trainer, "profile_diagnostics_mode"))
        extend = coordinate_profile_extension(
            Path(str(coordination_value)),
            arm=arm,
            candidate=candidate,
            diagnostics_mode=diagnostics_mode,
            step_walls_2_4=walls[1:4],
            threshold=float(_config_get(trainer, "profile_cv_threshold", 0.10)),
            timeout_seconds=float(_config_get(trainer, "profile_coordination_timeout_seconds", 900)),
        )
        return not extend

    def _compute_old_and_reference(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        policy_version = self._validate_rollout_policy_version(batch)
        _emit_audit_event(
            "boundary_return event=old_ref_start policy_version=%d mode=%s",
            policy_version,
            self._boundary_config().mode,
        )
        recorder = IntervalRecorder(f"boundary-old-ref-step-{self.global_steps}")
        with recorder.record("old_and_reference_log_prob"):
            batch = super()._compute_old_and_reference(batch, metrics, timing_raw)
        self._boundary_profile_intervals.extend(recorder.intervals)
        _emit_audit_event(
            "boundary_return event=old_ref_complete policy_version=%d mode=%s",
            policy_version,
            self._boundary_config().mode,
        )
        return batch

    def _compute_advantage_and_actor_update(
        self, batch: DataProto, metrics: dict[str, float], timing_raw: dict[str, float]
    ) -> tuple[DataProto, DataProto]:
        policy_version = self._validate_rollout_policy_version(batch)
        _emit_audit_event(
            "boundary_return event=grpo_actor_start policy_version=%d mode=%s",
            policy_version,
            self._boundary_config().mode,
        )
        recorder = IntervalRecorder(f"boundary-actor-step-{self.global_steps}")
        with recorder.record("advantage_and_actor_update"):
            result = super()._compute_advantage_and_actor_update(batch, metrics, timing_raw)
        self._boundary_profile_intervals.extend(recorder.intervals)
        _emit_audit_event(
            "boundary_return event=actor_update_complete policy_version=%d mode=%s",
            policy_version,
            self._boundary_config().mode,
        )
        return result
