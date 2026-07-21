import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

if "cachetools" not in sys.modules:
    cachetools = types.ModuleType("cachetools")

    class _LRUCache(dict):
        def __init__(self, maxsize):
            super().__init__()
            self.maxsize = maxsize

    cachetools.LRUCache = _LRUCache
    sys.modules["cachetools"] = cachetools

rollout_utils = types.ModuleType("verl.workers.rollout.utils")
rollout_utils.update_prometheus_config = lambda *_args, **_kwargs: None
sys.modules.setdefault("verl.workers.rollout.utils", rollout_utils)

from verl.experimental.probe_credit.probe_runtime import (
    ProbeBranchResult,
    ProbeTrajectory,
    aggregate_probe_results,
    aggregate_probe_successes,
    build_probe_requests,
    derive_grouped_request_seed,
    first_nonempty_line,
    immediate_verifier_text,
    generate_grouped_probe_results,
    relative_horizons,
)
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


POSITIONS = [0.0, 0.25, 0.5, 0.75, 0.9]


def _offline_module():
    path = Path(__file__).resolve().parents[3] / "tools/offline_long_reward_probe/forced_answer_probe.py"
    spec = importlib.util.spec_from_file_location("forced_answer_probe_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_horizons_use_floor_and_preserve_short_duplicates():
    assert relative_horizons(11, POSITIONS) == (0, 2, 5, 8, 9)
    assert relative_horizons(2, POSITIONS) == (0, 0, 1, 1, 1)


def test_candidate_and_verifier_protocol():
    assert first_nonempty_line("\n \n  42  \nignored") == "42"
    assert first_nonempty_line(" \n\t") == ""
    assert immediate_verifier_text("42") == "Answer: 42"


def test_build_requests_uses_raw_tokens_and_deduplicates_h0_and_short_prefixes():
    trajectories = [
        ProbeTrajectory("p", "a", (10, 11), (20, 21)),
        ProbeTrajectory("p", "b", (10, 11), (30, 31)),
    ]

    requests = build_probe_requests(
        trajectories,
        policy_version=7,
        relative_positions=POSITIONS,
        answer_prefix_token_ids=(90, 91),
        n=4,
        max_tokens=8,
        max_model_len=32,
    )

    assert len(requests) == 3  # one shared h0 plus one deduplicated nonzero prefix per trajectory
    h0 = next(request for request in requests if request.trajectory_id == "__prompt__")
    assert h0.input_token_ids == (10, 11, 90, 91)
    assert {(target.trajectory_index, target.position_indices) for target in h0.targets} == {
        (0, (0, 1)),
        (1, (0, 1)),
    }
    a_h1 = next(request for request in requests if request.trajectory_id == "a" and request.absolute_horizon == 1)
    assert a_h1.input_token_ids == (10, 11, 20, 90, 91)
    assert a_h1.targets[0].position_indices == (2, 3, 4)


def test_raw_prefix_matches_offline_probe_builder():
    offline = _offline_module()
    trajectory = ProbeTrajectory("p", "a", (1, 2), (3, 4, 5, 6))
    request = build_probe_requests(
        [trajectory],
        policy_version=1,
        relative_positions=[0.5],
        answer_prefix_token_ids=(7, 8),
        n=4,
        max_tokens=2,
        max_model_len=20,
        probe_zero_position=False,
    )[0]

    assert list(request.input_token_ids) == offline.build_probe_token_ids((1, 2), (3, 4, 5, 6), 2, (7, 8))


def test_grouped_seed_is_stable_and_uses_prompt_sentinel_for_h0():
    args = (5, "uid", "trajectory", 0.25, (0, 1, 2, 3))
    assert derive_grouped_request_seed(*args) == derive_grouped_request_seed(*args)
    assert derive_grouped_request_seed(*args) != derive_grouped_request_seed(6, *args[1:])


def test_aggregate_successes_is_branch_order_independent_and_strict():
    assert aggregate_probe_successes({2: 1.0, 0: 0.0, 3: 1.0, 1: 1.0}, n=4, strict=True) == 0.75
    with pytest.raises(ValueError, match="missing"):
        aggregate_probe_successes({0: 1.0}, n=4, strict=True)
    with pytest.raises(ValueError, match="branch"):
        aggregate_probe_successes({0: 1.0, 1: 0.0, 2: 1.0, 4: 1.0}, n=4, strict=True)


def test_aggregate_results_uses_ids_not_return_order_and_broadcasts_deduped_values():
    trajectories = [
        ProbeTrajectory("p", "a", (10,), (20, 21)),
        ProbeTrajectory("p", "b", (10,), (30, 31)),
    ]
    requests = build_probe_requests(
        trajectories,
        policy_version=9,
        relative_positions=POSITIONS,
        answer_prefix_token_ids=(90,),
        n=4,
        max_tokens=2,
        max_model_len=20,
    )
    results = []
    for request_index, request in enumerate(reversed(requests)):
        success = float(request_index % 2)
        for branch_id in (3, 1, 0, 2):
            results.append(ProbeBranchResult(request.request_id, branch_id, success))

    aggregate = aggregate_probe_results(requests, reversed(results), trajectory_count=2, position_count=5, n=4)

    assert aggregate.valid_mask == ((True,) * 5, (True,) * 5)
    for request in requests:
        expected = next(result.success for result in results if result.request_id == request.request_id)
        for target in request.targets:
            for position_index in target.position_indices:
                assert aggregate.values[target.trajectory_index][position_index] == expected


def test_strict_context_overflow_fails_without_truncation():
    with pytest.raises(ValueError, match="context overflow"):
        build_probe_requests(
            [ProbeTrajectory("p", "a", (1, 2, 3), (4, 5))],
            policy_version=1,
            relative_positions=[0.5],
            answer_prefix_token_ids=(6,),
            n=4,
            max_tokens=5,
            max_model_len=9,
            probe_zero_position=False,
            strict=True,
        )


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, **kwargs):
        return self.fn(**kwargs)


def test_llm_server_client_generate_grouped_uses_one_rpc_and_copies_sampling_params():
    calls = []
    server = type("Server", (), {})()
    server.generate_grouped = _RemoteMethod(
        lambda **kwargs: calls.append(kwargs)
        or [TokenOutput(token_ids=[branch], extra_fields={"text": str(branch), "branch_id": branch}) for branch in range(4)]
    )
    client = LLMServerClient(config={"actor_rollout_ref": {}}, load_balancer_handle=None)

    async def acquire(_request_id):
        return "server", server

    client._acquire_server = acquire
    client._release_server = lambda _server_id: None
    sampling = {"n": 4, "seed": 17, "max_tokens": 8, "stop": ["\n"]}

    outputs = asyncio.run(
        client.generate_grouped("logical", prompt_ids=[1, 2], sampling_params=sampling)
    )

    assert [output.extra_fields["branch_id"] for output in outputs] == [0, 1, 2, 3]
    assert len(calls) == 1
    assert calls[0]["sampling_params"] == sampling
    assert calls[0]["sampling_params"] is not sampling
    assert sampling == {"n": 4, "seed": 17, "max_tokens": 8, "stop": ["\n"]}


def test_probe_generator_uses_one_grouped_call_per_prefix_and_stable_branch_mapping():
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def generate_grouped(self, request_id, *, prompt_ids, sampling_params):
            self.calls.append((request_id, prompt_ids, sampling_params))
            return [
                TokenOutput(token_ids=[branch], extra_fields={"text": f"\n {branch} \n", "branch_id": branch})
                for branch in (3, 1, 0, 2)
            ]

    request = build_probe_requests(
        [ProbeTrajectory("p", "a", (1,), (2, 3, 4, 5))],
        policy_version=4,
        relative_positions=[0.5],
        answer_prefix_token_ids=(9,),
        n=4,
        max_tokens=8,
        max_model_len=32,
        probe_zero_position=False,
    )[0]
    client = FakeClient()
    base_sampling = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_tokens": 8, "stop": ["\n"]}

    results = generate_grouped_probe_results(
        client,
        [request],
        sampling_params=base_sampling,
        score_candidate=lambda _request, verifier_text: verifier_text in {"Answer: 1", "Answer: 3"},
    )

    assert len(client.calls) == 1
    assert client.calls[0][2]["n"] == 4
    assert client.calls[0][2]["seed"] == request.grouped_seed
    assert "n" not in base_sampling and "seed" not in base_sampling
    assert [(result.branch_id, result.success) for result in results] == [(3, True), (1, True), (0, False), (2, False)]
