import pytest
import torch

from verl.experimental.probe_credit.readiness_dominance import compute_readiness_dominance


def _compute(
    values,
    valid,
    *,
    terminal_success=None,
    positive=None,
    group_ids=None,
    n=4,
    strict_branch_margin=1,
    min_common_positions=2,
):
    batch_size = values.shape[0]
    if terminal_success is None:
        terminal_success = torch.ones(batch_size, dtype=torch.bool)
    if positive is None:
        positive = torch.ones(batch_size, dtype=torch.bool)
    if group_ids is None:
        group_ids = ["p"] * batch_size
    return compute_readiness_dominance(
        values,
        valid,
        terminal_success,
        positive,
        group_ids,
        n=n,
        strict_branch_margin=strict_branch_margin,
        min_common_positions=min_common_positions,
    )


def test_direct_dominance_uses_only_shared_active_horizons():
    values = torch.tensor([[0.50, 0.75, 1.00, 0.00], [0.25, 0.75, 0.75, 1.00]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])

    result, metrics = _compute(values, valid)

    assert result.dominance_matrix.tolist() == [[False, True], [False, False]]
    assert result.eligible_mask.tolist() == [True, True]
    assert result.frontier_mask.tolist() == [True, False]
    assert result.dominated_mask.tolist() == [False, True]
    assert result.group_has_dominance.tolist() == [True, True]
    assert metrics["dominance/eligible_group_rate"] == 1.0
    assert metrics["dominance/group_with_dominance_rate"] == 1.0
    assert metrics["dominance/eligible_positive_rate"] == 1.0
    assert metrics["dominance/dominated_positive_rate"] == 0.5
    assert metrics["dominance/frontier_fraction"] == 0.5
    assert metrics["dominance/comparable_pair_count"] == 1.0
    assert metrics["dominance/pair_coverage_rate"] == 1.0
    assert metrics["dominance/common_positions_mean"] == 3.0
    assert metrics["dominance/common_positions_min"] == 3.0
    assert metrics["dominance/common_positions_max"] == 3.0


def test_crossing_profiles_are_directly_nondominated_and_counted():
    values = torch.tensor([[0.25, 1.00, 0.50], [0.50, 0.75, 0.75]])
    valid = torch.ones_like(values, dtype=torch.bool)

    result, metrics = _compute(values, valid)

    assert not bool(result.dominance_matrix.any())
    assert result.frontier_mask.tolist() == [True, True]
    assert result.group_has_dominance.tolist() == [False, False]
    assert metrics["dominance/profile_cross_rate"] == 1.0
    assert metrics["dominance/group_with_dominance_rate"] == 0.0


def test_identical_profiles_do_not_dominate_or_count_as_crossing():
    values = torch.tensor([[0.25, 0.50, 0.75], [0.25, 0.50, 0.75]])
    valid = torch.ones_like(values, dtype=torch.bool)

    result, metrics = _compute(values, valid)

    assert not bool(result.dominance_matrix.any())
    assert result.frontier_mask.tolist() == [True, True]
    assert metrics["dominance/profile_cross_rate"] == 0.0


@pytest.mark.parametrize(
    ("terminal_success", "positive", "group_ids"),
    [
        (torch.tensor([True, True]), torch.tensor([True, True]), ["left", "right"]),
        (torch.tensor([True, False]), torch.tensor([True, True]), ["p", "p"]),
        (torch.tensor([True, True]), torch.tensor([True, False]), ["p", "p"]),
    ],
)
def test_ineligible_pairs_are_not_compared(terminal_success, positive, group_ids):
    values = torch.tensor([[0.75, 1.00], [0.25, 0.50]])
    valid = torch.ones_like(values, dtype=torch.bool)

    result, metrics = _compute(
        values,
        valid,
        terminal_success=terminal_success,
        positive=positive,
        group_ids=group_ids,
    )

    assert not bool(result.dominance_matrix.any())
    assert metrics["dominance/comparable_pair_count"] == 0.0
    assert metrics["dominance/eligible_group_rate"] == 0.0


def test_pair_with_insufficient_common_positions_is_not_compared():
    values = torch.tensor([[1.00, 0.75, 0.00], [0.50, 0.25, 1.00]])
    valid = torch.tensor([[True, True, False], [False, True, True]])

    result, metrics = _compute(values, valid)

    assert not bool(result.dominance_matrix.any())
    assert result.frontier_mask.tolist() == [True, True]
    assert metrics["dominance/same_group_eligible_pair_count"] == 1.0
    assert metrics["dominance/comparable_pair_count"] == 0.0
    assert metrics["dominance/pair_coverage_rate"] == 0.0


def test_multi_trajectory_group_builds_directly_nondominated_set():
    values = torch.tensor(
        [
            [0.75, 1.00, 1.00],
            [0.50, 0.75, 0.75],
            [1.00, 0.50, 1.00],
            [0.25, 0.25, 0.25],
        ]
    )
    valid = torch.ones_like(values, dtype=torch.bool)

    result, metrics = _compute(values, valid)

    assert result.dominance_matrix.tolist() == [
        [False, True, False, True],
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, False],
    ]
    assert result.frontier_mask.tolist() == [True, False, True, False]
    assert result.dominated_mask.tolist() == [False, True, False, True]
    assert metrics["dominance/frontier_size_mean"] == 2.0


def test_pair_specific_support_does_not_create_transitive_edge():
    values = torch.tensor(
        [
            [1.00, 1.00, 0.00, 0.00],
            [0.50, 0.50, 1.00, 1.00],
            [0.00, 0.00, 0.50, 0.50],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
            [False, False, True, True],
        ]
    )

    result, metrics = _compute(values, valid)

    assert result.dominance_matrix.tolist() == [
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ]
    assert metrics["dominance/same_group_eligible_pair_count"] == 3.0
    assert metrics["dominance/comparable_pair_count"] == 2.0
    assert metrics["dominance/pair_coverage_rate"] == pytest.approx(2 / 3)


def test_result_is_deterministic():
    values = torch.tensor([[0.75, 1.00], [0.25, 0.50], [0.50, 0.75]])
    valid = torch.ones_like(values, dtype=torch.bool)

    first, first_metrics = _compute(values, valid)
    second, second_metrics = _compute(values, valid)

    assert torch.equal(first.dominance_matrix, second.dominance_matrix)
    assert torch.equal(first.eligible_mask, second.eligible_mask)
    assert torch.equal(first.frontier_mask, second.frontier_mask)
    assert torch.equal(first.dominated_mask, second.dominated_mask)
    assert torch.equal(first.group_has_dominance, second.group_has_dominance)
    assert first_metrics == second_metrics


@pytest.mark.parametrize(
    ("values", "valid", "terminal_success", "positive", "group_ids", "message"),
    [
        (
            torch.zeros(2, 2, 1),
            torch.ones(2, 2, 1, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            "probe_values",
        ),
        (
            torch.zeros(2, 2),
            torch.ones(2, 3, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            "same shape",
        ),
        (
            torch.zeros(2, 2, dtype=torch.long),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            "floating",
        ),
        (
            torch.zeros(2, 2),
            torch.ones(2, 2),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            "valid_mask",
        ),
        (
            torch.zeros(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            "terminal_success",
        ),
        (
            torch.zeros(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2),
            ["p", "p"],
            "positive_trajectory_mask",
        ),
        (
            torch.tensor([[0.0, float("nan")], [0.0, 1.0]]),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            "finite",
        ),
        (
            torch.tensor([[0.0, 1.1], [0.0, 1.0]]),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p", "p"],
            r"\[0, 1\]",
        ),
        (
            torch.zeros(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            ["p"],
            "group_ids",
        ),
    ],
)
def test_invalid_tensor_inputs_fail_closed(
    values, valid, terminal_success, positive, group_ids, message
):
    with pytest.raises(ValueError, match=message):
        compute_readiness_dominance(
            values,
            valid,
            terminal_success,
            positive,
            group_ids,
            n=4,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 0}, "n"),
        ({"strict_branch_margin": 0}, "strict_branch_margin"),
        ({"min_common_positions": 0}, "min_common_positions"),
    ],
)
def test_invalid_protocol_parameters_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _compute(
            torch.zeros(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            **kwargs,
        )
