# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in, read-only vanilla GRPO batch audit. No RNG calls or tensor writes."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def audit_batch(batch, config, step, use_reference_policy):
    tensors = batch.batch
    uids = list(batch.non_tensor_batch["uid"])
    counts = Counter(uids)
    assert len(counts) == 256 and set(counts.values()) == {8}, counts
    assert len(uids) == 2048
    assert not use_reference_policy and "ref_log_prob" not in tensors
    assert "rollout_is_weights" not in tensors
    rewards = tensors["token_level_rewards"].detach().cpu()
    scores = tensors["token_level_scores"].detach().cpu()
    advantages = tensors["advantages"].detach().cpu()
    mask = tensors["response_mask"].detach().cpu()
    assert torch.equal(rewards, scores), "reward shaping detected"
    totals = rewards.sum(-1)
    assert torch.isfinite(rewards).all() and torch.isfinite(advantages).all()
    assert set(totals.tolist()) <= {-1.0, 1.0}, "unexpected math score"
    lengths = mask.sum(-1)
    assert lengths.min() > 0 and lengths.max() <= 2048
    mixed = same = 0
    for uid in counts:
        rows = [i for i, key in enumerate(uids) if key == uid]
        values = totals[rows]
        expected = (values - values.mean()) / (values.std() + 1e-6)
        assert torch.allclose(advantages[rows], expected[:, None] * mask[rows], atol=1e-6)
        if values.min() == values.max():
            same += 1
            assert torch.count_nonzero(advantages[rows]) == 0
        else:
            mixed += 1
    digest = hashlib.sha256()
    for key in ["input_ids", "responses", "attention_mask"]:
        value = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    row = dict(
        step=step,
        groups=len(counts),
        trajectories=len(uids),
        mixed_groups=mixed,
        same_reward_groups=same,
        reward_mean=totals.mean().item(),
        advantage_abs_max=advantages.abs().max().item(),
        max_length=lengths.max().item(),
        input_order_sha256=digest.hexdigest(),
        uid_order_sha256=hashlib.sha256(json.dumps(uids).encode()).hexdigest(),
        reference_enabled=use_reference_policy,
        status="PASS",
    )
    path = Path(config.trainer.grpo_audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def write_progress(config, step, metrics):
    import math
    import time
    import numpy as np
    path = Path(config.trainer.grpo_audit_path).parent / "progress"
    path.mkdir(exist_ok=True)
    data = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            data[key] = float(value)
    assert all(math.isfinite(v) for v in data.values()), "nonfinite training metrics"
    row = {"step": step, "time": time.time(), "metrics": data}
    target = path / f"step_{step:04d}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(target)
