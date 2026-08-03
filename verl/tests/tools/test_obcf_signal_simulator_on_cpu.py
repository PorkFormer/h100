from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from tools.on_policy_budgeted_capability_floor.simulate_obcf_signal import (
    ENGINEERING_GATE_THRESHOLD,
    engineering_exit_code,
    simulate_obcf_signal,
)

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 2048


def _cache_rows():
    values = ((0, 2, 0.125), (1, 4, 0.375), (2, 8, 0.875), (3, 2, 0.125))
    return [
        {
            "prompt_key": f"key-{prompt_id}",
            "prompt_id": prompt_id,
            "prompt_hash": f"hash-{prompt_id}",
            "base_rollout_count": 8,
            "base_prefix_success_count": base_count,
            "floor_count": round(floor * 8),
            "capability_floor": floor,
        }
        for prompt_id, base_count, floor in values
    ]


def _score_rows(*, include_full=False):
    rewards = {
        0: (False, True),
        1: (False, False),
        2: (False, True),
        3: (True, True),
        99: (False, True),
    }
    rows = []
    for prompt_id, prompt_rewards in rewards.items():
        for rollout_index, reward in enumerate(prompt_rewards):
            row = {
                "model_id": "current",
                "prompt_id": prompt_id,
                "prompt_hash": f"hash-{prompt_id}",
                "rollout_index": rollout_index,
                f"prefix_reward_{BUDGET}": reward,
                f"prefix_error_{BUDGET}": None,
                f"prefix_token_count_{BUDGET}": 10,
                "response_token_count": 10,
            }
            if include_full:
                # Prompt 1 is delayed; all prefix-supported prompts are retained.
                row["full_reward"] = prompt_id != 99 and (prompt_id != 1 or rollout_index == 0)
                row["full_error"] = None
            rows.append(row)
    return rows


def _simulate(rows=None):
    return simulate_obcf_signal(
        cache_rows=_cache_rows(),
        current_score_rows=_score_rows() if rows is None else rows,
        reference_budget=BUDGET,
        current_rollouts_per_prompt=2,
        train_batch_size=10,
        model_id="current",
    )


def test_synthetic_report_has_exact_counts_metrics_and_engineering_gate():
    report = _simulate()

    assert report["protected_prompt_count"] == 4
    assert report["current_prompt_count"] == 5
    assert report["q_current_mean"] == pytest.approx(0.5)
    assert report["floor_mean"] == pytest.approx(0.375)
    assert report["deficit_mean"] == pytest.approx(0.1875)
    assert report["active_group_fraction"] == pytest.approx(0.5)
    assert report["mixed_group_fraction"] == pytest.approx(0.5)
    assert report["all_zero_group_fraction"] == pytest.approx(0.25)
    assert report["all_one_group_fraction"] == pytest.approx(0.25)
    assert report["nonzero_gradient_fraction"] == pytest.approx(0.25)
    assert report["active_without_gradient_fraction"] == pytest.approx(0.25)
    assert report["expected_prefix_verifier_calls_per_step"] == pytest.approx(16.0)
    strata = report["by_base_success_count"]
    assert sum(value["prompt_count"] for value in strata.values()) == 4
    assert strata["2"]["prompt_count"] == 2
    assert strata["2"]["deficit_mean"] == pytest.approx(0.0)
    assert strata["4"]["deficit_mean"] == pytest.approx(0.375)
    assert strata["8"]["deficit_mean"] == pytest.approx(0.375)
    assert report["actionability"] == {
        "protected_prompt_count": 4,
        "actionable_prompt_count": 1,
        "inert_prompt_count": 3,
        "inert_prompt_fraction": pytest.approx(0.75),
        "passed": False,
    }
    gate = report["engineering_gates"]
    assert gate["signal_density"]["threshold"] == ENGINEERING_GATE_THRESHOLD == 0.05
    assert gate["signal_density"]["passed"] is True
    assert gate["actionability"]["passed"] is False
    assert gate["passed"] is False


def test_optional_retained_delayed_lost_breakdown_is_prompt_level():
    report = _simulate(_score_rows(include_full=True))

    assert report["retained_delayed_lost"] == {
        "retained": {"count": 3, "fraction": pytest.approx(0.75)},
        "delayed": {"count": 1, "fraction": pytest.approx(0.25)},
        "lost": {"count": 0, "fraction": pytest.approx(0.0)},
    }


def test_all_zero_active_groups_fail_the_engineering_gate_honestly():
    rows = _score_rows()
    for row in rows:
        if row["prompt_id"] in {0, 1, 2, 3}:
            row[f"prefix_reward_{BUDGET}"] = False
    report = _simulate(rows)

    assert report["nonzero_gradient_fraction"] == 0.0
    assert report["active_without_gradient_fraction"] == 1.0
    assert report["engineering_gates"]["signal_density"]["passed"] is False


def test_zero_token_mixed_group_has_no_gradient_and_cannot_pass_gate():
    rows = _score_rows()
    for row in rows:
        if row["prompt_id"] == 2:
            row[f"prefix_token_count_{BUDGET}"] = 0
            row["response_token_count"] = 0
        elif row["prompt_id"] in {0, 1, 3}:
            row[f"prefix_reward_{BUDGET}"] = False
    report = _simulate(rows)
    assert report["nonzero_gradient_fraction"] == 0.0
    assert report["engineering_gates"]["signal_density"]["passed"] is False


def _single_prompt_simulation(*, base_success_count, tolerance, rewards):
    n0 = 8
    rows = [
        {
            "prompt_id": 0,
            "prompt_hash": "hash-0",
            "base_rollout_count": n0,
            "base_prefix_success_count": base_success_count,
            "floor_count": base_success_count - 1,
            "capability_floor": (base_success_count - 1) / n0,
        }
    ]
    scores = [
        {
            "model_id": "current",
            "prompt_id": 0,
            "prompt_hash": "hash-0",
            "rollout_index": index,
            f"prefix_reward_{BUDGET}": reward,
            f"prefix_error_{BUDGET}": None,
            f"prefix_token_count_{BUDGET}": 10,
            "response_token_count": 10,
        }
        for index, reward in enumerate(rewards)
    ]
    return simulate_obcf_signal(
        cache_rows=rows,
        current_score_rows=scores,
        reference_budget=BUDGET,
        current_rollouts_per_prompt=len(rewards),
        train_batch_size=1,
        model_id="current",
        override_reference_tolerance_count=tolerance,
    )


def test_tolerance_override_stratifies_actionable_and_inert_floors():
    inert = _single_prompt_simulation(
        base_success_count=2,
        tolerance=1,
        rewards=[False, True, False, False, False, False, False, False],
    )
    assert inert["actionability"]["inert_prompt_count"] == 1
    assert inert["by_base_success_count"]["2"]["structurally_inert_fraction"] == 1.0

    actionable = _single_prompt_simulation(
        base_success_count=2,
        tolerance=0,
        rewards=[True, False, False, False, False, False, False, False],
    )
    stratum = actionable["by_base_success_count"]["2"]
    assert stratum["capability_floor"] == pytest.approx(2 / 8)
    assert stratum["active_fraction"] == 1.0
    assert stratum["mixed_fraction"] == 1.0
    assert stratum["nonzero_gradient_fraction"] == 1.0

    base_three = _single_prompt_simulation(
        base_success_count=3,
        tolerance=1,
        rewards=[True, False, False, False, False, False, False, False],
    )
    assert base_three["by_base_success_count"]["3"]["active_fraction"] == 1.0


def test_active_all_zero_reports_deficit_without_gradient():
    report = _single_prompt_simulation(
        base_success_count=2,
        tolerance=0,
        rewards=[False] * 8,
    )
    stratum = report["by_base_success_count"]["2"]
    assert stratum["deficit_mean"] == pytest.approx(2 / 8)
    assert stratum["all_zero_fraction"] == 1.0
    assert stratum["nonzero_gradient_fraction"] == 0.0
    assert stratum["active_without_gradient_fraction"] == 1.0


def test_override_does_not_mutate_cache_rows_and_exit_codes_are_distinct():
    cache_rows = _cache_rows()
    before = copy.deepcopy(cache_rows)
    report = simulate_obcf_signal(
        cache_rows=cache_rows,
        current_score_rows=_score_rows(),
        reference_budget=BUDGET,
        current_rollouts_per_prompt=2,
        train_batch_size=10,
        model_id="current",
        override_reference_tolerance_count=0,
    )
    assert cache_rows == before
    assert report["floor_configuration"]["source"] == "override"
    assert engineering_exit_code(report) == 3

    passing = copy.deepcopy(report)
    passing["engineering_gates"]["actionability"]["passed"] = True
    passing["engineering_gates"]["signal_density"]["passed"] = True
    assert engineering_exit_code(passing) == 0
    passing["engineering_gates"]["signal_density"]["passed"] = False
    assert engineering_exit_code(passing) == 2


def test_retained_delayed_lost_is_a_complete_nonmonotonic_partition():
    rows = _score_rows(include_full=True)
    for row in rows:
        if row["prompt_id"] == 0:
            row["full_reward"] = False  # prefix success remains retained
        if row["prompt_id"] == 1:
            row["full_reward"] = False  # no prefix and no full success is lost
    report = _simulate(rows)
    breakdown = report["retained_delayed_lost"]
    assert breakdown["retained"]["count"] == 3
    assert breakdown["delayed"]["count"] == 0
    assert breakdown["lost"]["count"] == 1
    assert sum(value["count"] for value in breakdown.values()) == 4


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate"),
        (lambda rows: rows.pop(0), "rollout indices|count"),
        (lambda rows: rows[0].update(prompt_hash="wrong"), "prompt_hash"),
        (lambda rows: rows[0].pop(f"prefix_reward_{BUDGET}"), "prefix_reward"),
        (lambda rows: rows[0].update({f"prefix_reward_{BUDGET}": 1}), "boolean"),
        (lambda rows: rows[0].update({f"prefix_error_{BUDGET}": "boom"}), "prefix_error"),
    ],
)
def test_malformed_or_unmatched_current_groups_fail_closed(mutation, match):
    rows = _score_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        _simulate(rows)


def test_missing_protected_prompt_and_mixed_model_ids_fail_closed():
    rows = [row for row in _score_rows() if row["prompt_id"] != 2]
    with pytest.raises(ValueError, match="protected prompt"):
        _simulate(rows)

    rows = _score_rows()
    rows[0]["model_id"] = "other"
    with pytest.raises(ValueError, match="model_id"):
        _simulate(rows)


def test_module_help_is_available():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.on_policy_budgeted_capability_floor.simulate_obcf_signal",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "engineering gate" in result.stdout.lower()

    malformed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.on_policy_budgeted_capability_floor.simulate_obcf_signal",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert malformed.returncode == 1
