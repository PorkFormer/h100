"""Optimizer-agnostic NCBR reward hook; actor tensors never enter long scoring."""
from __future__ import annotations

import contextlib
import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BoundaryRewardOutput,
    apply_boundary_return,
)
from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations
from verl.utils.debug import marked_timer


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

@contextlib.contextmanager
def preserve_driver_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


@dataclass
class NCBROutcome:
    effective_scores: torch.Tensor
    boundary_mask: torch.Tensor | None
    corrected_mask: torch.Tensor | None
    metrics: dict
    aux: dict | None


class NCBRHook:
    def __init__(self, *, continuation_runner=run_boundary_continuations, reward_adapter=apply_boundary_return):
        self._continue = continuation_runner
        self._adapt = reward_adapter

    def apply(
        self, batch: DataProto, raw_scores: torch.Tensor, *, raw_verifier_extras: dict,
        config: Any, policy_version: int, sampling_params: dict,
        continuation_client: Any, score_long: Callable, eos_token_id: int | None,
        short_response_length: int, max_model_len: int, timing_raw: dict | None = None,
        long_reward_complete: Callable | None = None,
    ) -> NCBROutcome:
        if getattr(config, "enable", None) is True:
            config.validate()
        if getattr(config, "enable", None) is False or config.mode == "off":
            return NCBROutcome(raw_scores, None, None, {}, None)
        config.validate()
        agents = batch.non_tensor_batch.get("agent_name")
        if agents is not None and any(str(agent) != "single_turn_agent" for agent in agents):
            raise ValueError("boundary_return v1 requires every row to use single_turn_agent")
        if _contains_multimodal_payload(batch):
            raise ValueError("boundary_return v1 does not support multimodal input rows")
        timing_raw = {} if timing_raw is None else timing_raw
        with preserve_driver_rng_state():
            # Neither the adapter nor a verifier callback can mutate the caller's batch.
            work = copy.deepcopy(batch)
            work.batch["token_level_scores"] = raw_scores.clone()
            work.batch["token_level_rewards"] = raw_scores.clone()
            work.non_tensor_batch.update(copy.deepcopy(raw_verifier_extras))
            with marked_timer("boundary_continuation", timing_raw, color="magenta"):
                capture = self._continue(
                    config=config, rollout_batch=work, client=continuation_client,
                    eos_token_id=eos_token_id, short_response_length=short_response_length,
                    max_model_len=max_model_len, policy_version=policy_version,
                    sampling_params=sampling_params,
                )
            if capture is None:
                raise AssertionError("active boundary_return returned no continuation capture")
            if capture.generations:
                with marked_timer("boundary_long_reward", timing_raw, color="magenta"):
                    long_reward = score_long(work, capture.generations, config)
                if long_reward_complete is not None:
                    long_reward_complete(policy_version, len(capture.generations))
            else:
                long_reward = BoundaryRewardOutput(
                    reward_tensor=torch.empty((0, 0), dtype=torch.float32),
                    extra_info={config.correctness_key: np.asarray([], dtype=np.float64),
                                config.task_score_key: np.asarray([], dtype=np.float64)},
                )
            result = self._adapt(work, capture=capture, long_reward_output=long_reward, config=config,
                                 include_group_statistics=False)
            effective = work.batch["token_level_scores"]
            if (effective.shape, effective.dtype, effective.device) != (
                raw_scores.shape, raw_scores.dtype, raw_scores.device
            ):
                raise AssertionError("NCBR changed token-score shape, dtype or device")
            changed = (effective != raw_scores).any(dim=-1)
            return NCBROutcome(
                effective, torch.as_tensor(capture.hit_response_cap, dtype=torch.bool, device=raw_scores.device),
                changed, result.metrics,
                {"result": result, "capture": capture,
                 "row_labels": {k: v for k, v in work.batch.items() if k.startswith("boundary_")},
                 "reward_extras": {k: v for k, v in work.non_tensor_batch.items() if k.startswith("boundary_")}},
            )
