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
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.natural_continuation_boundary_return.runtime import (
    BoundaryContinuationRequest,
    aggregate_continuation_results,
    build_continuation_requests,
    derive_continuation_seed,
    run_boundary_continuations,
)
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.workers.rollout.replica import TokenOutput


def _batch(*, finish_reasons=("length", "stop"), versions=(7, 7)) -> DataProto:
    prompts = torch.tensor([[0, 11, 12], [0, 21, 22]], dtype=torch.long)
    responses = torch.tensor([[31, 32, 33, 34], [41, 42, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]], dtype=torch.long
    )
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention_mask,
            "position_ids": torch.clamp(attention_mask.cumsum(-1) - 1, min=0),
        },
        non_tensors={
            "uid": np.asarray(["shared", "shared"], dtype=object),
            "trajectory_id": np.asarray(["shared:0", "shared:1"], dtype=object),
            "finish_reason": np.asarray(finish_reasons, dtype=object),
            "rollout_policy_version": np.asarray(versions, dtype=object),
        },
    )


def _active_config(**overrides) -> BoundaryReturnConfig:
    values = {
        "mode": "shadow",
        "long_response_length": 6,
        "max_concurrent_requests": 2,
        "request_batch_size": 2,
        "seed": 19,
    }
    values.update(overrides)
    return BoundaryReturnConfig(**values)


def _normal_sampling() -> dict[str, object]:
    return {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": -1,
    }


def test_boundary_return_typed_config_defaults_to_off_and_validates_strictly():
    config = BoundaryReturnConfig()
    assert config.mode == "off"
    assert config.correctness_key == "acc"
    assert config.task_score_key == "score"
    assert config.correctness_threshold == 0.5
    config.validate()

    for kwargs, message in (
        ({"mode": "bonus"}, "mode"),
        ({"long_response_length": 0}, "long_response_length"),
        ({"correctness_key": ""}, "correctness_key"),
        ({"task_score_key": ""}, "task_score_key"),
        ({"correctness_threshold": float("nan")}, "correctness_threshold"),
        ({"max_concurrent_requests": 0}, "max_concurrent_requests"),
        ({"request_batch_size": 1, "max_concurrent_requests": 2}, "request_batch_size"),
        ({"seed": True}, "seed"),
        ({"strict": False}, "strict"),
    ):
        with pytest.raises(ValueError, match=message):
            BoundaryReturnConfig(**kwargs).validate()


def test_build_requests_uses_exact_prompt_and_prefix_and_trajectory_identity():
    requests, hit_cap = build_continuation_requests(
        _batch(finish_reasons=("length", "length")),
        policy_version=7,
        short_response_length=4,
        long_response_length=6,
        max_model_len=8,
        eos_token_id=99,
        base_seed=19,
    )

    assert hit_cap.tolist() == [True, True]
    assert [request.input_token_ids for request in requests] == [
        (11, 12, 31, 32, 33, 34),
        (21, 22, 41, 42),
    ]
    assert [request.max_tokens for request in requests] == [2, 4]
    assert requests[0].routing_key == requests[1].routing_key
    assert requests[0].request_id != requests[1].request_id
    assert requests[0].seed != requests[1].seed
    assert requests[0].branch_count == 1
    assert requests == build_continuation_requests(
        _batch(finish_reasons=("length", "length")),
        policy_version=7,
        short_response_length=4,
        long_response_length=6,
        max_model_len=8,
        eos_token_id=99,
        base_seed=19,
    )[0]
    assert derive_continuation_seed(19, 7, "shared", "shared:0") == requests[0].seed


def test_build_requests_rejects_context_shortage_and_bad_identity():
    with pytest.raises(ValueError, match="context"):
        build_continuation_requests(
            _batch(),
            policy_version=7,
            short_response_length=4,
            long_response_length=6,
            max_model_len=7,
            eos_token_id=99,
            base_seed=0,
        )

    batch = _batch(finish_reasons=("length", "length"))
    batch.non_tensor_batch["trajectory_id"] = np.asarray(["same", "same"], dtype=object)
    with pytest.raises(ValueError, match="duplicate trajectory"):
        build_continuation_requests(
            batch,
            policy_version=7,
            short_response_length=4,
            long_response_length=6,
            max_model_len=8,
            eos_token_id=99,
            base_seed=0,
        )


class _Client:
    def __init__(self, factory=None):
        self.calls = []
        self.factory = factory

    async def generate_grouped(self, request_id, *, prompt_ids, sampling_params, routing_key):
        self.calls.append((request_id, prompt_ids, dict(sampling_params), routing_key))
        if self.factory is not None:
            return await self.factory(request_id, prompt_ids, sampling_params, routing_key)
        return [
            SimpleNamespace(
                token_ids=[],
                extra_fields={"branch_id": 0, "global_steps": 7},
            )
        ]


def test_disabled_mode_returns_before_detection_or_client_access():
    client = _Client()
    capture = run_boundary_continuations(
        config=BoundaryReturnConfig(mode="off"),
        rollout_batch=object(),
        client=client,
        eos_token_id=99,
        short_response_length=4,
        max_model_len=8,
        policy_version=7,
        sampling_params=_normal_sampling(),
    )
    assert capture is None
    assert client.calls == []


def test_zero_token_tail_is_legal_and_sampling_matches_normal_rollout():
    client = _Client()
    capture = run_boundary_continuations(
        config=_active_config(),
        rollout_batch=_batch(),
        client=client,
        eos_token_id=99,
        short_response_length=4,
        max_model_len=8,
        policy_version=7,
        sampling_params=_normal_sampling(),
    )

    assert capture.hit_response_cap.tolist() == [True, False]
    assert len(capture.generations) == 1
    assert capture.generations[0].tail_token_ids == ()
    assert capture.generations[0].prefix_token_ids == (31, 32, 33, 34)
    assert client.calls[0][2] == {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": -1,
        "n": 1,
        "seed": capture.requests[0].seed,
        "max_tokens": 2,
    }


def _request(identity: str, trajectory: str) -> BoundaryContinuationRequest:
    return BoundaryContinuationRequest(
        parent_index=0,
        request_id=identity,
        routing_key="route",
        policy_version=7,
        uid="u",
        trajectory_id=trajectory,
        prompt_token_ids=(1, 2),
        prefix_token_ids=(3, 4),
        input_token_ids=(1, 2, 3, 4),
        max_tokens=2,
        seed=1,
    )


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([], "missing"),
        (
            [
                SimpleNamespace(request_id="r", branch_id=0, tail_token_ids=(), actual_policy_version=7),
                SimpleNamespace(request_id="r", branch_id=0, tail_token_ids=(), actual_policy_version=7),
            ],
            "duplicate",
        ),
        (
            [SimpleNamespace(request_id="r", branch_id=1, tail_token_ids=(), actual_policy_version=7)],
            "branch",
        ),
        (
            [SimpleNamespace(request_id="r", branch_id=0, tail_token_ids=(), actual_policy_version=None)],
            "missing actual policy",
        ),
        (
            [SimpleNamespace(request_id="r", branch_id=0, tail_token_ids=(), actual_policy_version=6)],
            "does not match",
        ),
    ],
)
def test_result_mapping_fails_closed(results, message):
    with pytest.raises(ValueError, match=message):
        aggregate_continuation_results([_request("r", "t")], results, expected_policy_version=7)


def test_result_mapping_rejects_mixed_versions_across_requests():
    results = [
        SimpleNamespace(request_id="r1", branch_id=0, tail_token_ids=(), actual_policy_version=7),
        SimpleNamespace(request_id="r2", branch_id=0, tail_token_ids=(), actual_policy_version=8),
    ]
    with pytest.raises(ValueError, match="mixed actual policy"):
        aggregate_continuation_results(
            [_request("r1", "t1"), _request("r2", "t2")],
            results,
            expected_policy_version=7,
        )


def test_concurrent_failure_cancels_sibling_tasks_and_preserves_global_rng():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def factory(request_id, _prompt_ids, _sampling_params, _routing_key):
        if len(client.calls) == 1:
            await started.wait()
            raise RuntimeError("boom")
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    client = _Client(factory)
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="boom"):
        run_boundary_continuations(
            config=_active_config(),
            rollout_batch=_batch(finish_reasons=("length", "length")),
            client=client,
            eos_token_id=99,
            short_response_length=4,
            max_model_len=8,
            policy_version=7,
            sampling_params=_normal_sampling(),
        )
    assert cancelled.is_set()
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "preserved"), [("off", False), ("shadow", True), ("replace", True)])
async def test_single_turn_preserves_finish_reason_only_when_boundary_return_is_active(mode, preserved):
    class Server:
        async def generate(self, **_kwargs):
            return TokenOutput(token_ids=[1, 2], log_probs=None, stop_reason="length", extra_fields={})

    loop = object.__new__(SingleTurnAgentLoop)
    loop.prompt_length = 4
    loop.response_length = 4
    loop.rollout_config = {
        "forced_answer_probe": {"enable": False},
        "boundary_return": {"mode": mode},
    }
    loop.server_manager = Server()

    async def process(_messages):
        return {}

    async def tokenize(_messages, **_kwargs):
        return [10, 11]

    loop.process_multi_modal_info = process
    loop.apply_chat_template = tokenize
    loop._get_mm_processor_kwargs = lambda _audios: None

    output = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "x"}])
    assert ("finish_reason" in output.extra_fields) is preserved
    if preserved:
        assert output.extra_fields["finish_reason"] == "length"
