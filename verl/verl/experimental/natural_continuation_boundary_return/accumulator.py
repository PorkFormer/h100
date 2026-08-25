"""Optimizer-step aggregation across every dynamic-sampling candidate batch."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from verl.experimental.natural_continuation_boundary_return.reward_adapter import (
    BoundaryReturnBatchResult,
)


class BoundaryReturnStepAccumulator:
    """Pool row observations so counts, denominators, and quantiles are step-global."""

    def __init__(self, *, correctness_threshold: float) -> None:
        self.correctness_threshold = float(correctness_threshold)
        self._batches: list[BoundaryReturnBatchResult] = []

    def add(self, result: BoundaryReturnBatchResult) -> None:
        self._batches.append(result)

    def metrics(self) -> dict[str, float]:
        if not self._batches:
            return {}
        hit = np.concatenate([batch.hit_response_cap for batch in self._batches])
        short = np.concatenate([batch.short_acc for batch in self._batches])
        long = np.concatenate([batch.long_acc for batch in self._batches])
        boundary = np.concatenate([batch.boundary_acc for batch in self._batches])
        deltas = np.concatenate([batch.task_score_delta for batch in self._batches])
        tails = np.concatenate([batch.tail_token_lengths for batch in self._batches])
        uids = np.concatenate([batch.uids for batch in self._batches])
        valid = hit & np.isfinite(long)
        short_success = short >= self.correctness_threshold
        long_success = long >= self.correctness_threshold
        cap_failure = valid & ~short_success
        cap_success = valid & short_success
        recovered = cap_failure & long_success
        regressed = cap_success & ~long_success
        tail_values = tails[hit]
        normal_tokens = sum(batch.normal_response_tokens for batch in self._batches)

        grouped_short: dict[str, list[float]] = defaultdict(list)
        grouped_boundary: dict[str, list[float]] = defaultdict(list)
        for uid, short_value, boundary_value in zip(uids.tolist(), short.tolist(), boundary.tolist(), strict=True):
            grouped_short[str(uid)].append(float(short_value))
            grouped_boundary[str(uid)].append(float(boundary_value))
        all_wrong = [
            uid
            for uid, values in grouped_short.items()
            if values and all(value < self.correctness_threshold for value in values)
        ]
        unlocked = [
            uid
            for uid in all_wrong
            if np.ptp(np.asarray(grouped_boundary[uid], dtype=np.float64)) > 0.0
        ]

        metrics = {
            "boundary_return/candidate_count": float(len(hit)),
            "boundary_return/hit_cap_count": float(hit.sum()),
            "boundary_return/hit_cap_rate": float(hit.mean()) if len(hit) else 0.0,
            "boundary_return/valid_long_score_count": float(valid.sum()),
            "boundary_return/long_success_rate_given_cap": (
                float(long_success[valid].mean()) if valid.any() else 0.0
            ),
            "boundary_return/recovered_count": float(recovered.sum()),
            "boundary_return/recovered_rate_given_cap_failure": (
                float(recovered.sum() / cap_failure.sum()) if cap_failure.any() else 0.0
            ),
            "boundary_return/regressed_count": float(regressed.sum()),
            "boundary_return/regressed_rate_given_cap_success": (
                float(regressed.sum() / cap_success.sum()) if cap_success.any() else 0.0
            ),
            "boundary_return/extra_generated_tokens": float(tail_values.sum()),
            "boundary_return/extra_generated_token_ratio": (
                float(tail_values.sum() / normal_tokens) if normal_tokens else 0.0
            ),
            "boundary_return/tail_tokens_mean": float(tail_values.mean()) if len(tail_values) else 0.0,
            "boundary_return/tail_tokens_p50": (
                float(np.percentile(tail_values, 50)) if len(tail_values) else 0.0
            ),
            "boundary_return/tail_tokens_p90": (
                float(np.percentile(tail_values, 90)) if len(tail_values) else 0.0
            ),
            "boundary_return/task_score_delta_mean_given_cap": (
                float(deltas[hit].mean()) if hit.any() else 0.0
            ),
            "boundary_return/task_score_delta_min_given_cap": (
                float(deltas[hit].min()) if hit.any() else 0.0
            ),
            "boundary_return/task_score_delta_max_given_cap": (
                float(deltas[hit].max()) if hit.any() else 0.0
            ),
            "boundary_return/short_all_wrong_group_count": float(len(all_wrong)),
            "boundary_return/unlocked_group_count": float(len(unlocked)),
            "boundary_return/unlocked_group_rate": (
                float(len(unlocked) / len(all_wrong)) if all_wrong else 0.0
            ),
            "boundary_return/prefix_penalty_drift_max": float(
                max(
                    batch.metrics.get("boundary_return/prefix_penalty_drift_max", 0.0)
                    for batch in self._batches
                )
            ),
        }
        transition_masks = {
            "h_wrong_l_wrong": valid & ~short_success & ~long_success,
            "h_wrong_l_correct": valid & ~short_success & long_success,
            "h_correct_l_correct": valid & short_success & long_success,
            "h_correct_l_wrong": valid & short_success & ~long_success,
        }
        valid_count = int(valid.sum())
        for name, mask in transition_masks.items():
            metrics[f"boundary_return/transition_{name}_count"] = float(mask.sum())
            metrics[f"boundary_return/transition_{name}_rate"] = (
                float(mask.sum() / valid_count) if valid_count else 0.0
            )
        return metrics
