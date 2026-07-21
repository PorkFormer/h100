from collections import defaultdict

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.probe_credit.dynamic_sampling import (
    filter_dapo_generation_batch,
    select_complete_prompt_groups,
)


def _batch(uids, metrics, trajectory_ids=None):
    if trajectory_ids is None:
        trajectory_ids = [f"t{i}" for i in range(len(uids))]
    return DataProto.from_dict(
        tensors={"dummy": torch.arange(len(uids)).unsqueeze(-1)},
        non_tensors={
            "uid": np.asarray(uids, dtype=object),
            "acc": np.asarray(metrics, dtype=float),
            "trajectory_id": np.asarray(trajectory_ids, dtype=object),
        },
    )


def _literal_upstream_filter(batch, metric_name):
    """Literal selection block from verl-recipe e477deff97b15d067f8b4e71f75b80ae58ad64c4."""
    prompt_uid2metric_vals = defaultdict(list)
    for uid, metric_val in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True):
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
        idx for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"])
        if traj_from_prompt_uid in kept_prompt_uids
    ]
    return batch[kept_traj_idxs]


def test_filter_matches_upstream_positive_std_degenerate_and_singleton_semantics():
    batch = _batch(
        ["var", "var", "flat", "flat", "single"],
        [0.0, 1.0, 0.5, 0.5, 0.0],
    )

    actual = filter_dapo_generation_batch(batch, "acc")
    expected = _literal_upstream_filter(batch, "acc")

    assert actual.non_tensor_batch["trajectory_id"].tolist() == expected.non_tensor_batch["trajectory_id"].tolist()
    assert actual.non_tensor_batch["uid"].tolist() == ["var", "var", "single"]


def test_accumulation_and_complete_group_selection_match_upstream_prefix_slice():
    first = filter_dapo_generation_batch(_batch(["a"] * 4 + ["x"] * 4, [0, 1, 0, 1] + [0] * 4), "acc")
    second = filter_dapo_generation_batch(_batch(["b"] * 4 + ["c"] * 4, [0, 1, 0, 1] * 2), "acc")
    accumulated = DataProto.concat([first, second])

    selected = select_complete_prompt_groups(accumulated, prompt_count=2, rollout_n=4)

    assert selected.non_tensor_batch["uid"].tolist() == ["a"] * 4 + ["b"] * 4
    assert selected.non_tensor_batch["trajectory_id"].tolist() == accumulated[:8].non_tensor_batch[
        "trajectory_id"
    ].tolist()


def test_complete_group_selection_rejects_incomplete_or_insufficient_groups():
    with pytest.raises(ValueError, match="complete"):
        select_complete_prompt_groups(_batch(["a"] * 4 + ["b"] * 3, range(7)), prompt_count=2, rollout_n=4)
    with pytest.raises(ValueError, match="prompt groups"):
        select_complete_prompt_groups(_batch(["a"] * 4, range(4)), prompt_count=2, rollout_n=4)


def test_filter_does_not_accept_or_read_probe_configuration():
    batch = _batch(["a", "a"], [0.0, 1.0])
    with pytest.raises(TypeError):
        filter_dapo_generation_batch(batch, "acc", probe_credit={"enable": True})
