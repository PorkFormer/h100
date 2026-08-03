import pytest

from verl.experimental.success_support_floor.analysis import (
    binary_auroc,
    paired_bootstrap_mean_difference,
    spearman_correlation,
)


def test_binary_auroc_handles_perfect_order_and_ties():
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert binary_auroc([0.5, 0.5], [False, True]) == 0.5


def test_spearman_uses_rank_correlation():
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_paired_bootstrap_is_deterministic_and_reports_interval():
    first = paired_bootstrap_mean_difference([2, 4, 6], [1, 1, 1], seed=7, resamples=200)
    second = paired_bootstrap_mean_difference([2, 4, 6], [1, 1, 1], seed=7, resamples=200)
    assert first == second
    assert first["mean_difference"] == pytest.approx(3.0)
    assert first["ci_low"] <= first["mean_difference"] <= first["ci_high"]


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (binary_auroc, ([1, 2], [True, True])),
        (spearman_correlation, ([1], [2])),
        (paired_bootstrap_mean_difference, ([1], [1])),
    ],
)
def test_statistics_fail_closed_on_degenerate_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)
