import sys
import types

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

from verl.experimental.probe_credit.probe_runtime import (  # noqa: E402
    ProbeBranchResult,
    ProbeTrajectory,
    aggregate_probe_results,
    build_absolute_probe_requests,
    build_probe_requests,
    derive_absolute_grouped_request_seed,
    derive_grouped_request_seed,
)


def _build(trajectories, **overrides):
    kwargs = {
        "trajectory_mask": [True] * len(trajectories),
        "policy_version": 7,
        "absolute_horizons": [1, 3, 4],
        "answer_prefix_token_ids": (99,),
        "n": 4,
        "max_tokens": 2,
        "max_model_len": 32,
        "strict": True,
    }
    kwargs.update(overrides)
    return build_absolute_probe_requests(trajectories, **kwargs)


def test_absolute_plan_uses_same_token_budget_and_marks_inactive_cells():
    plan = _build(
        [
            ProbeTrajectory("p", "long", (10,), (20, 21, 22, 23, 24)),
            ProbeTrajectory("p", "short", (10,), (30, 31, 32)),
        ]
    )

    assert plan.absolute_horizons == (1, 3, 4)
    assert plan.valid_mask == ((True, True, True), (True, False, False))
    assert len(plan.requests) == 4
    assert all(request.position_kind == "absolute" for request in plan.requests)
    assert all(request.relative_position is None for request in plan.requests)
    assert {request.absolute_horizon for request in plan.requests} == {1, 3, 4}
    horizon_one = [request for request in plan.requests if request.absolute_horizon == 1]
    assert [request.input_token_ids for request in horizon_one] == [
        (10, 20, 99),
        (10, 30, 99),
    ]


def test_terminal_failure_and_inactive_horizons_create_no_requests():
    plan = _build(
        [
            ProbeTrajectory("p", "success", (10,), (20, 21, 22)),
            ProbeTrajectory("p", "failure", (10,), (30, 31, 32, 33, 34)),
        ],
        trajectory_mask=[True, False],
        absolute_horizons=[1, 2, 3],
    )

    assert plan.valid_mask == ((True, True, False), (False, False, False))
    assert [(request.trajectory_id, request.absolute_horizon) for request in plan.requests] == [
        ("success", 1),
        ("success", 2),
    ]


def test_absolute_request_ids_and_seeds_are_stable_and_namespaced():
    trajectory = ProbeTrajectory("p", "t", (10,), (20, 21, 22, 23, 24))

    first = _build([trajectory], absolute_horizons=[1]).requests[0]
    second = _build([trajectory], absolute_horizons=[1]).requests[0]
    relative = build_probe_requests(
        [trajectory],
        policy_version=7,
        relative_positions=[0.2],
        answer_prefix_token_ids=(99,),
        n=4,
        max_tokens=2,
        max_model_len=32,
        probe_zero_position=False,
    )[0]

    assert first.request_id == second.request_id
    assert first.grouped_seed == second.grouped_seed
    assert first.request_id != relative.request_id
    assert first.grouped_seed != relative.grouped_seed
    assert relative.position_kind == "relative"
    assert relative.relative_position == 0.2
    assert derive_grouped_request_seed(5, "uid", "trajectory", 0.25, (0, 1, 2, 3)) == 1274110086
    assert first.grouped_seed == derive_absolute_grouped_request_seed(
        7, "p", "t", 1, (0, 1, 2, 3)
    )


def test_absolute_aggregation_restores_values_and_planned_validity():
    trajectories = [
        ProbeTrajectory("p", "long", (10,), (20, 21, 22, 23, 24)),
        ProbeTrajectory("p", "short", (10,), (30, 31, 32)),
    ]
    plan = _build(trajectories)
    results = []
    for request in reversed(plan.requests):
        success = request.absolute_horizon / 4
        for branch_id in (3, 1, 0, 2):
            results.append(
                ProbeBranchResult(
                    request.request_id,
                    branch_id,
                    success,
                    actual_policy_version=7,
                )
            )

    aggregate = aggregate_probe_results(
        plan.requests,
        reversed(results),
        trajectory_count=2,
        position_count=3,
        n=4,
        strict=True,
        expected_policy_version=7,
    )

    assert aggregate.valid_mask == plan.valid_mask
    assert aggregate.values == ((0.25, 0.75, 1.0), (0.25, 0.0, 0.0))


def test_absolute_aggregation_rejects_mixed_or_missing_policy_versions_and_branches():
    plan = _build(
        [ProbeTrajectory("p", "t", (10,), (20, 21, 22))],
        absolute_horizons=[1],
    )
    request = plan.requests[0]

    with pytest.raises(ValueError, match="mixed actual policy versions"):
        aggregate_probe_results(
            plan.requests,
            [
                ProbeBranchResult(
                    request.request_id,
                    branch_id,
                    True,
                    actual_policy_version=7 if branch_id < 3 else 8,
                )
                for branch_id in range(4)
            ],
            trajectory_count=1,
            position_count=1,
            n=4,
            expected_policy_version=7,
        )

    with pytest.raises(ValueError, match="missing Probe branches"):
        aggregate_probe_results(
            plan.requests,
            [
                ProbeBranchResult(request.request_id, 0, True, actual_policy_version=7),
                ProbeBranchResult(request.request_id, 1, True, actual_policy_version=7),
            ],
            trajectory_count=1,
            position_count=1,
            n=4,
            strict=True,
            expected_policy_version=7,
        )


def test_absolute_context_overflow_fails_closed_without_truncation():
    with pytest.raises(ValueError, match="context overflow"):
        _build(
            [ProbeTrajectory("p", "t", (1, 2, 3), (4, 5, 6))],
            absolute_horizons=[2],
            answer_prefix_token_ids=(7,),
            max_tokens=5,
            max_model_len=10,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"absolute_horizons": []}, "absolute_horizons"),
        ({"absolute_horizons": [0, 1]}, "positive"),
        ({"absolute_horizons": [1, 1]}, "increasing"),
        ({"absolute_horizons": [2, 1]}, "increasing"),
        ({"absolute_horizons": [1, 2.5]}, "integer"),
        ({"trajectory_mask": []}, "trajectory_mask"),
        ({"trajectory_mask": [1]}, "boolean"),
        ({"n": 0}, "n"),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_absolute_planner_rejects_invalid_protocol_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        _build(
            [ProbeTrajectory("p", "t", (10,), (20, 21, 22))],
            **overrides,
        )


def test_absolute_planner_rejects_inconsistent_prompt_tokens_in_uid():
    with pytest.raises(ValueError, match="inconsistent prompt"):
        _build(
            [
                ProbeTrajectory("p", "a", (10,), (20, 21)),
                ProbeTrajectory("p", "b", (11,), (30, 31)),
            ],
            absolute_horizons=[1],
        )
