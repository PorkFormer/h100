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
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import build_rollout_sampling_params
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.natural_continuation_boundary_return.profiling import analyze_interval_dag
from verl.experimental.natural_continuation_boundary_return.runtime import (
    BoundaryContinuationBranchResult,
    BoundaryContinuationRequest,
    aggregate_continuation_results,
    build_continuation_requests,
    derive_continuation_seed,
    run_boundary_continuations,
)
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
from verl.workers.rollout.vllm_rollout.utils import (
    extract_request_engine_timing,
    request_engine_timing_diagnostics,
)


def _batch(*, finish_reasons=("length", "stop"), versions=(7, 7)) -> DataProto:
    prompts = torch.tensor([[0, 11, 12], [0, 21, 22]], dtype=torch.long)
    responses = torch.tensor([[31, 32, 33, 34], [41, 42, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
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


def test_normal_and_continuation_share_one_sampling_parameter_builder():
    rollout = SimpleNamespace(
        temperature=0.9,
        top_p=0.95,
        top_k=-1,
        calculate_log_probs=False,
        val_kwargs=SimpleNamespace(temperature=0.0, top_p=1.0, top_k=-1),
    )
    assert build_rollout_sampling_params(rollout) == {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": False,
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
        ({"request_timeout_seconds": 0}, "request_timeout_seconds"),
        ({"long_reward_chunk_size": 0}, "long_reward_chunk_size"),
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
    assert (
        requests
        == build_continuation_requests(
            _batch(finish_reasons=("length", "length")),
            policy_version=7,
            short_response_length=4,
            long_response_length=6,
            max_model_len=8,
            eos_token_id=99,
            base_seed=19,
        )[0]
    )
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

    async def start_grouped(self, request_id, *, prompt_ids, sampling_params, routing_key):
        self.calls.append((request_id, prompt_ids, dict(sampling_params), routing_key))
        client = self

        class Tracked:
            backend_request_id = request_id
            server_id = "server"

            async def result(self):
                if client.factory is not None:
                    return await client.factory(request_id, prompt_ids, sampling_params, routing_key)
                return [
                    SimpleNamespace(
                        token_ids=[],
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


def test_active_mode_rejects_untracked_grouped_client():
    with pytest.raises(TypeError, match="tracked grouped-request"):
        run_boundary_continuations(
            config=_active_config(),
            rollout_batch=_batch(),
            client=object(),
            eos_token_id=99,
            short_response_length=4,
            max_model_len=8,
            policy_version=7,
            sampling_params=_normal_sampling(),
        )


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
    assert capture.continuation_input_token_lengths == (6,)
    assert capture.long_hit_response_cap.tolist() == [False, False]
    assert capture.request_timeout_seconds == 600.0
    assert client.calls[0][2] == {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": -1,
        "n": 1,
        "seed": capture.requests[0].seed,
        "max_tokens": 2,
    }


def test_vllm_epoch_timestamps_are_mapped_into_the_monotonic_interval_clock():
    async def factory(_request_id, _prompt_ids, _sampling_params, _routing_key):
        await asyncio.sleep(0.01)
        finished = time.time()
        return [
            SimpleNamespace(
                token_ids=[88],
                stop_reason="completed",
                extra_fields={
                    "branch_id": 0,
                    "global_steps": 7,
                    "finish_reason": "stop",
                    "engine_timing": {
                        "arrival_time": finished - 0.003,
                        "first_scheduled_time": finished - 0.002,
                        "first_token_time": finished - 0.001,
                        "finished_time": finished,
                    },
                },
            )
        ]

    capture = run_boundary_continuations(
        config=_active_config(),
        rollout_batch=_batch(),
        client=_Client(factory=factory),
        eos_token_id=99,
        short_response_length=4,
        max_model_len=8,
        policy_version=7,
        sampling_params=_normal_sampling(),
    )
    analysis = analyze_interval_dag(capture.profiling_intervals)
    assert analysis["valid"], analysis["errors"]
    timestamps = [
        value for interval in capture.profiling_intervals for value in (interval.wall_start, interval.wall_end)
    ]
    assert max(timestamps) - min(timestamps) < 10.0


def test_vllm_request_metrics_normalize_legacy_and_current_clock_domains():
    legacy = SimpleNamespace(
        arrival_time=100.0,
        first_scheduled_time=101.0,
        first_token_time=102.0,
        finished_time=103.0,
    )
    assert extract_request_engine_timing(legacy) == {
        "clock_domain": "epoch",
        "arrival_time": 100.0,
        "first_scheduled_time": 101.0,
        "first_token_time": 102.0,
        "finished_time": 103.0,
    }

    current = SimpleNamespace(
        arrival_time=1_000_000.0,
        queued_ts=200.0,
        scheduled_ts=201.0,
        first_token_ts=202.0,
        last_token_ts=203.0,
    )
    assert extract_request_engine_timing(current) == {
        "clock_domain": "monotonic",
        "arrival_time": 200.0,
        "first_scheduled_time": 201.0,
        "first_token_time": 202.0,
        "finished_time": 203.0,
    }

    assert extract_request_engine_timing(
        {"queued_ts": 300.0, "scheduled_ts": 301.0, "first_token_ts": 302.0, "last_token_ts": 303.0}
    ) == {
        "clock_domain": "monotonic",
        "arrival_time": 300.0,
        "first_scheduled_time": 301.0,
        "first_token_time": 302.0,
        "finished_time": 303.0,
    }


def test_vllm_request_metrics_diagnostics_are_json_safe_and_complete():
    diagnostics = request_engine_timing_diagnostics(
        SimpleNamespace(queued_ts=0.0, scheduled_ts=2.0, first_token_ts=3.0, last_token_ts=4.0)
    )
    assert diagnostics["metrics_type"] == "types.SimpleNamespace"
    assert diagnostics["fields"]["queued_ts"] == 0.0
    assert diagnostics["fields"]["scheduled_ts"] == 2.0
    assert diagnostics["fields"]["arrival_time"] is None
    assert set(diagnostics["fields"]) == {
        "arrival_time",
        "first_scheduled_time",
        "first_token_time",
        "finished_time",
        "queued_ts",
        "scheduled_ts",
        "first_token_ts",
        "last_token_ts",
    }


def test_vllm_grouped_generation_propagates_current_engine_timing():
    class FakeEngine:
        async def generate(self, **_kwargs):
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[17, 18], finish_reason="stop", text="ok")],
                metrics={
                    "queued_ts": 300.0,
                    "scheduled_ts": 301.0,
                    "first_token_ts": 302.0,
                    "last_token_ts": 303.0,
                },
            )

    server = object.__new__(vLLMHttpServer)
    server.config = OmegaConf.create(
        {
            "max_model_len": 16,
            "repetition_penalty": 1.0,
            "forced_answer_probe": {"enable": False},
            "boundary_return": {"mode": "replace"},
            "enable_rollout_routing_replay": False,
            "mtp": None,
        }
    )
    server.model_config = OmegaConf.create(
        {"processor": None, "lora_rank": 0, "lora": {"rank": 0, "merge": False}}
    )
    server.engine = FakeEngine()
    server.global_steps = 7

    outputs = asyncio.run(
        server.generate_grouped(
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 2, "n": 1, "temperature": 1.0},
            request_id="grouped-timing-test",
        )
    )
    assert len(outputs) == 1
    assert outputs[0].extra_fields["global_steps"] == 7
    assert outputs[0].extra_fields["engine_timing"] == {
        "clock_domain": "monotonic",
        "arrival_time": 300.0,
        "first_scheduled_time": 301.0,
        "first_token_time": 302.0,
        "finished_time": 303.0,
    }


def test_vllm_monotonic_timestamps_preserve_engine_intervals():
    async def factory(_request_id, _prompt_ids, _sampling_params, _routing_key):
        await asyncio.sleep(0.01)
        finished = time.perf_counter()
        return [
            SimpleNamespace(
                token_ids=[88],
                stop_reason="completed",
                extra_fields={
                    "branch_id": 0,
                    "global_steps": 7,
                    "finish_reason": "stop",
                    "engine_timing": {
                        "clock_domain": "monotonic",
                        "arrival_time": finished - 0.003,
                        "first_scheduled_time": finished - 0.002,
                        "first_token_time": finished - 0.001,
                        "finished_time": finished,
                    },
                },
            )
        ]

    capture = run_boundary_continuations(
        config=_active_config(),
        rollout_batch=_batch(),
        client=_Client(factory=factory),
        eos_token_id=99,
        short_response_length=4,
        max_model_len=8,
        policy_version=7,
        sampling_params=_normal_sampling(),
    )
    intervals = {interval.name: interval for interval in capture.profiling_intervals}
    assert intervals["continuation_queue"].wall_end - intervals["continuation_queue"].wall_start == pytest.approx(
        0.001
    )
    assert intervals["continuation_prefill_engine"].wall_end - intervals[
        "continuation_prefill_engine"
    ].wall_start == pytest.approx(0.001)
    assert intervals["continuation_decode_engine"].wall_end - intervals[
        "continuation_decode_engine"
    ].wall_start == pytest.approx(0.001)


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
    for result in results:
        result.stop_reason = "completed"
        result.finish_reason = "stop"
    with pytest.raises(ValueError, match=message):
        aggregate_continuation_results([_request("r", "t")], results, expected_policy_version=7)


def test_result_mapping_rejects_mixed_versions_across_requests():
    results = [
        SimpleNamespace(
            request_id="r1",
            branch_id=0,
            tail_token_ids=(),
            actual_policy_version=7,
            stop_reason="completed",
            finish_reason="stop",
        ),
        SimpleNamespace(
            request_id="r2",
            branch_id=0,
            tail_token_ids=(),
            actual_policy_version=8,
            stop_reason="completed",
            finish_reason="stop",
        ),
    ]
    with pytest.raises(ValueError, match="mixed actual policy"):
        aggregate_continuation_results(
            [_request("r1", "t1"), _request("r2", "t2")],
            results,
            expected_policy_version=7,
        )


@pytest.mark.parametrize("bad_reason", ["abort", "aborted", "error", "cancelled", "timeout", None])
def test_strict_result_mapping_rejects_invalid_terminal_states(bad_reason):
    result = BoundaryContinuationBranchResult(
        request_id="r",
        branch_id=0,
        tail_token_ids=(9,),
        actual_policy_version=7,
        stop_reason=bad_reason,
        finish_reason="stop",
    )
    with pytest.raises(ValueError, match="terminal"):
        aggregate_continuation_results([_request("r", "t")], [result], expected_policy_version=7)


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason", "accepted"),
    [
        ("completed", "stop", True),
        ("eos", "eos", True),
        ("stop", "stop", True),
        ("completed", "length", True),
        ("length", "length", False),
    ],
)
def test_zero_token_tail_requires_legal_terminal_state(stop_reason, finish_reason, accepted):
    result = BoundaryContinuationBranchResult(
        request_id="r",
        branch_id=0,
        tail_token_ids=(),
        actual_policy_version=7,
        stop_reason=stop_reason,
        finish_reason=finish_reason,
    )
    if accepted:
        aggregate_continuation_results([_request("r", "t")], [result], expected_policy_version=7)
    else:
        with pytest.raises(ValueError, match="zero-token"):
            aggregate_continuation_results([_request("r", "t")], [result], expected_policy_version=7)


def test_result_mapping_rejects_tail_larger_than_request_budget():
    result = BoundaryContinuationBranchResult(
        request_id="r",
        branch_id=0,
        tail_token_ids=(9, 10, 11),
        actual_policy_version=7,
        stop_reason="completed",
        finish_reason="length",
    )
    with pytest.raises(ValueError, match="max_tokens"):
        aggregate_continuation_results([_request("r", "t")], [result], expected_policy_version=7)


def test_remote_abort_drain_release_happens_before_local_settle_and_preserves_primary_error(capsys):
    events = []
    both_started = asyncio.Event()

    class Tracked:
        def __init__(self, request_id, fail):
            self.request_id = request_id
            self.backend_request_id = f"backend-{request_id}"
            self.server_id = "server"
            self.server = object()
            self.object_ref = object()
            self.fail = fail

        async def result(self):
            events.append(f"result:{self.request_id}")
            if len([event for event in events if event.startswith("result:")]) == 2:
                both_started.set()
            await both_started.wait()
            if self.fail:
                raise RuntimeError("primary boom")
            try:
                await asyncio.sleep(60)
            finally:
                events.append(f"settled:{self.request_id}")

        async def abort(self):
            events.append(f"abort:{self.request_id}")
            if self.fail:
                raise RuntimeError("cleanup abort boom")

        async def drain(self):
            events.append("drain:server")

        async def release(self):
            events.append(f"release:{self.request_id}")

    class TrackedClient:
        def __init__(self):
            self.count = 0

        async def start_grouped(self, request_id, **_kwargs):
            self.count += 1
            return Tracked(request_id, fail=self.count == 1)

    with pytest.raises(RuntimeError, match="primary boom") as caught:
        run_boundary_continuations(
            config=_active_config(),
            rollout_batch=_batch(finish_reasons=("length", "length")),
            client=TrackedClient(),
            eos_token_id=99,
            short_response_length=4,
            max_model_len=8,
            policy_version=7,
            sampling_params=_normal_sampling(),
        )

    abort_positions = [index for index, event in enumerate(events) if event.startswith("abort:")]
    drain_position = events.index("drain:server")
    release_positions = [index for index, event in enumerate(events) if event.startswith("release:")]
    settle_positions = [index for index, event in enumerate(events) if event.startswith("settled:")]
    assert abort_positions and max(abort_positions) < drain_position
    assert release_positions and drain_position < min(release_positions)
    assert settle_positions and max(release_positions) < min(settle_positions)
    assert any("cleanup abort boom" in note for note in getattr(caught.value, "__notes__", []))
    audit = capsys.readouterr().out
    assert audit.index("event=abort_start") < audit.index("event=drain_start")
    assert audit.index("event=drain_ack") < audit.index("event=release_start")
    assert audit.index("event=release_ack") < audit.index("event=local_settle")


def test_timeout_shields_backend_result_until_explicit_remote_abort(capsys):
    events = []
    remote_done = asyncio.Event()

    class Tracked:
        backend_request_id = "backend-timeout"
        server_id = "server"
        server = object()
        object_ref = object()
        active = False

        async def result(self):
            self.active = True
            events.append("result_started")
            try:
                await remote_done.wait()
                events.append("result_finished")
                return []
            except asyncio.CancelledError:
                events.append("result_cancelled")
                raise
            finally:
                self.active = False

        async def abort(self):
            events.append(f"abort_active:{self.active}")
            if not self.active:
                raise RuntimeError("remote request was cancelled before explicit abort")
            remote_done.set()
            return {"mechanism": "ray_object_ref_cancel"}

        async def drain(self):
            events.append("drain")
            await remote_done.wait()

        async def release(self):
            events.append("release")

    class Client:
        async def start_grouped(self, _request_id, **_kwargs):
            return Tracked()

    with pytest.raises(TimeoutError):
        run_boundary_continuations(
            config=_active_config(
                max_concurrent_requests=1,
                request_batch_size=1,
                request_timeout_seconds=0.01,
            ),
            rollout_batch=_batch(),
            client=Client(),
            eos_token_id=99,
            short_response_length=4,
            max_model_len=8,
            policy_version=7,
            sampling_params=_normal_sampling(),
        )

    assert "abort_active:True" in events
    assert events.index("abort_active:True") < events.index("drain") < events.index("release")
    assert "result_cancelled" not in events
    assert "event=abort_ack count=1 errors=0 mechanisms=['ray_object_ref_cancel']" in capsys.readouterr().out


def test_remote_cleanup_waits_for_every_delayed_start_before_result_failure():
    events = []

    class Tracked:
        def __init__(self, label, fail):
            self.label = label
            self.backend_request_id = f"backend-{label}"
            self.server_id = "server"
            self.server = object()
            self.object_ref = object()
            self.fail = fail

        async def result(self):
            events.append(f"result:{self.label}")
            if self.fail:
                raise RuntimeError("primary boom")
            await asyncio.sleep(60)

        async def abort(self):
            events.append(f"abort:{self.label}")

        async def drain(self):
            events.append("drain:server")

        async def release(self):
            events.append(f"release:{self.label}")

    class Client:
        def __init__(self):
            self.count = 0

        async def start_grouped(self, _request_id, **_kwargs):
            self.count += 1
            label = str(self.count)
            events.append(f"start:{label}")
            if label == "2":
                await asyncio.sleep(0.01)
            events.append(f"started:{label}")
            return Tracked(label, fail=label == "1")

    with pytest.raises(RuntimeError, match="primary boom"):
        run_boundary_continuations(
            config=_active_config(),
            rollout_batch=_batch(finish_reasons=("length", "length")),
            client=Client(),
            eos_token_id=99,
            short_response_length=4,
            max_model_len=8,
            policy_version=7,
            sampling_params=_normal_sampling(),
        )

    first_result = events.index("result:1")
    assert events.index("started:2") < first_result
    assert {event for event in events if event.startswith("abort:")} == {"abort:1", "abort:2"}
    drain_position = events.index("drain:server")
    assert max(events.index("abort:1"), events.index("abort:2")) < drain_position
    assert drain_position < min(events.index("release:1"), events.index("release:2"))


def test_per_request_timeout_remotely_aborts_and_exposes_timeout_audit_fields():
    events = []

    class Tracked:
        request_id = "r"
        backend_request_id = "backend-r"
        server_id = "server"
        server = object()
        object_ref = object()

        async def result(self):
            await asyncio.sleep(60)

        async def abort(self):
            events.append("abort")

        async def drain(self):
            events.append("drain")

        async def release(self):
            events.append("release")

    class Client:
        async def start_grouped(self, *_args, **_kwargs):
            return Tracked()

    with pytest.raises(TimeoutError) as caught:
        run_boundary_continuations(
            config=_active_config(request_timeout_seconds=0.01),
            rollout_batch=_batch(),
            client=Client(),
            eos_token_id=99,
            short_response_length=4,
            max_model_len=8,
            policy_version=7,
            sampling_params=_normal_sampling(),
        )
    assert events == ["abort", "drain", "release"]
    assert caught.value.boundary_continuation_timeout_count == 1
    assert caught.value.boundary_continuation_request_timeout_seconds == 0.01


def test_concurrent_failure_cancels_sibling_tasks_and_preserves_global_rng():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    factory_call_count = 0

    async def factory(request_id, _prompt_ids, _sampling_params, _routing_key):
        nonlocal factory_call_count
        factory_call_count += 1
        if factory_call_count == 1:
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
