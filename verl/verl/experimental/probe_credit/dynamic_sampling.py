"""Pure helpers preserving official DAPO Dynamic Sampling selection semantics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from verl import DataProto


def filter_dapo_generation_batch(batch: DataProto, metric_name: str) -> DataProto:
    """Keep full prompt groups with positive metric std, matching verl-recipe e477deff."""
    prompt_uid2metric_vals: dict[object, list[object]] = defaultdict(list)
    for uid, metric_val in zip(
        batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
    ):
        prompt_uid2metric_vals[uid].append(metric_val)

    prompt_uid2metric_std = {
        prompt_uid: np.std(metric_vals) for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
    }
    kept_prompt_uids = [
        uid
        for uid, std in prompt_uid2metric_std.items()
        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
    ]
    kept_traj_idxs = [
        idx
        for idx, trajectory_uid in enumerate(batch.non_tensor_batch["uid"])
        if trajectory_uid in kept_prompt_uids
    ]
    return batch[kept_traj_idxs]


def select_complete_prompt_groups(batch: DataProto, prompt_count: int, rollout_n: int) -> DataProto:
    """Select the first requested complete prompt groups without splitting trajectories."""
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive")
    if rollout_n <= 0:
        raise ValueError("rollout_n must be positive")

    ordered_uids = list(dict.fromkeys(batch.non_tensor_batch["uid"].tolist()))
    if len(ordered_uids) < prompt_count:
        raise ValueError(f"need {prompt_count} prompt groups, found {len(ordered_uids)}")
    selected_uids = ordered_uids[:prompt_count]
    selected_indices = [
        index for index, uid in enumerate(batch.non_tensor_batch["uid"])
        if uid in selected_uids
    ]
    counts = {uid: 0 for uid in selected_uids}
    for index in selected_indices:
        counts[batch.non_tensor_batch["uid"][index]] += 1
    incomplete = {uid: count for uid, count in counts.items() if count != rollout_n}
    if incomplete:
        raise ValueError(f"selected prompt groups are not complete rollout_n={rollout_n} groups: {incomplete}")
    if len(selected_indices) != prompt_count * rollout_n:
        raise ValueError("complete prompt-group selection produced an unexpected trajectory count")
    return batch[selected_indices]
