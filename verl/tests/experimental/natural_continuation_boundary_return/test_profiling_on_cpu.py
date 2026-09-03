from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from verl.experimental.natural_continuation_boundary_return.profiling import (
    IntervalRecorder,
    ProfileInterval,
    analyze_interval_dag,
    coefficient_of_variation,
    coordinate_profile_extension,
    interval_union_seconds,
    normalized_unit_cost,
)


def test_interval_union_does_not_sum_concurrent_request_latency():
    intervals = [
        ProfileInterval("a", "decode", 0.0, 4.0, None, True),
        ProfileInterval("b", "decode", 1.0, 3.0, None, True),
        ProfileInterval("c", "decode", 5.0, 7.0, None, True),
    ]
    assert interval_union_seconds(intervals) == 6.0


def test_nested_timer_uses_child_union_for_exclusive_time_and_fails_closed_on_u_fixed():
    intervals = [
        ProfileInterval("step", "step", 0.0, 10.0, None, False),
        ProfileInterval("request-a", "request", 1.0, 5.0, "step", True),
        ProfileInterval("request-b", "request", 3.0, 7.0, "step", True),
    ]
    analysis = analyze_interval_dag(
        intervals,
        full_step_interval_id="step",
        modeled_interval_ids=("request-a", "request-b"),
    )
    assert analysis["valid"]
    assert analysis["exclusive_seconds"]["step"] == 4.0
    assert analysis["u_fixed"] == 4.0
    unavailable = analyze_interval_dag(intervals)
    assert unavailable["u_fixed"] == "unavailable"
    assert unavailable["u_fixed_unavailable_reason"] == "full_step_coverage_unavailable"


def test_recorder_emits_required_interval_provenance():
    recorder = IntervalRecorder("test")
    with recorder.record("parent") as parent_id:
        with recorder.record("child", asynchronous=True):
            time.sleep(0.001)
    by_name = {interval.name: interval for interval in recorder.intervals}
    assert by_name["child"].parent_id == parent_id
    assert by_name["child"].asynchronous
    assert by_name["child"].wall_end >= by_name["child"].wall_start


def test_cost_helpers():
    assert normalized_unit_cost(10, 5) == 2
    assert normalized_unit_cost(10, 0) == "unavailable"
    assert coefficient_of_variation([10.0, 10.0, 10.0]) == 0
    assert coefficient_of_variation([0.0, 0.0]) == "unavailable"
    assert coefficient_of_variation([9.0, 10.0, 11.0]) == pytest.approx((2 / 3) ** 0.5 / 10)


def test_paired_profile_coordination_extends_both_live_arms_when_either_is_unstable(tmp_path):
    def coordinate(arm, walls):
        return coordinate_profile_extension(
            tmp_path,
            arm=arm,
            candidate="P1",
            diagnostics_mode="on",
            step_walls_2_4=walls,
            timeout_seconds=2,
            poll_seconds=0.01,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline = executor.submit(coordinate, "baseline", [10.0, 10.0, 10.0])
        v1 = executor.submit(coordinate, "v1", [8.0, 10.0, 12.0])
    assert baseline.result()
    assert v1.result()
