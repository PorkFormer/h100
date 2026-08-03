"""Dependency-light paired statistics for the offline BSSF surrogate gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with at least two values")
    return array


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def binary_auroc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    score = _finite_vector(scores, "scores")
    labels = np.asarray(positive, dtype=np.bool_)
    if labels.ndim != 1 or labels.shape != score.shape:
        raise ValueError("positive labels must match scores")
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both classes")
    rank_sum = float(_average_ranks(score)[labels].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = _finite_vector(left, "left")
    right_array = _finite_vector(right, "right")
    if left_array.shape != right_array.shape:
        raise ValueError("paired vectors must have equal length")
    left_rank = _average_ranks(left_array)
    right_rank = _average_ranks(right_array)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.sqrt((left_rank**2).sum() * (right_rank**2).sum()))
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for constant ranks")
    return float((left_rank * right_rank).sum() / denominator)


def paired_bootstrap_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = 20260803,
    resamples: int = 10000,
) -> dict[str, float | int]:
    left_array = _finite_vector(left, "left")
    right_array = _finite_vector(right, "right")
    if left_array.shape != right_array.shape:
        raise ValueError("paired vectors must have equal length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    difference = left_array - right_array
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(resamples, len(difference)))
    estimates = difference[indices].mean(axis=1)
    return {
        "prompt_count": len(difference),
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "seed": seed,
        "resamples": resamples,
    }
