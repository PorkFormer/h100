# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import math

import pytest
import torch

from verl.trainer.ppo.core_algos import get_current_clip_ratios, linear_schedule
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, ClipRatioScheduleConfig
from verl.workers.utils.losses import ppo_loss


def _actor_config(**kwargs):
    defaults = {
        "strategy": "fsdp",
        "rollout_n": 1,
        "use_dynamic_bsz": True,
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
    }
    defaults.update(kwargs)
    return ActorConfig(**defaults)


def test_linear_schedule_returns_start_at_step_zero():
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=10, global_step=0) == pytest.approx(0.35)


def test_linear_schedule_interpolates_midpoint():
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=10, global_step=5) == pytest.approx(0.275)


def test_linear_schedule_returns_end_at_and_after_schedule_steps():
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=10, global_step=10) == pytest.approx(0.2)
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=10, global_step=25) == pytest.approx(0.2)


def test_linear_schedule_zero_steps_returns_end():
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=0, global_step=0) == pytest.approx(0.2)


def test_linear_schedule_honors_start_step():
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=10, global_step=4, start_step=5) == pytest.approx(0.35)
    assert linear_schedule(start=0.35, end=0.2, schedule_steps=10, global_step=10, start_step=5) == pytest.approx(
        0.275
    )


def test_get_current_clip_ratios_returns_static_values_when_schedule_disabled():
    config = _actor_config(
        clip_ratio_low=0.1,
        clip_ratio_high=0.3,
        clip_ratio_schedule=ClipRatioScheduleConfig(
            enable=False,
            clip_low_start=0.2,
            clip_low_end=0.2,
            clip_high_start=0.5,
            clip_high_end=0.2,
            schedule_steps=10,
        ),
    )

    assert get_current_clip_ratios(config, global_step=5) == pytest.approx((0.1, 0.3))


def test_get_current_clip_ratios_falls_back_to_static_values_for_missing_bounds():
    config = _actor_config(
        clip_ratio_low=0.1,
        clip_ratio_high=0.3,
        clip_ratio_schedule=ClipRatioScheduleConfig(enable=True, schedule_steps=10),
    )

    assert get_current_clip_ratios(config, global_step=5) == pytest.approx((0.1, 0.3))


def test_get_current_clip_ratios_interpolates_enabled_linear_schedule():
    config = _actor_config(
        clip_ratio_schedule=ClipRatioScheduleConfig(
            enable=True,
            clip_low_start=0.2,
            clip_low_end=0.2,
            clip_high_start=0.35,
            clip_high_end=0.2,
            schedule_steps=10,
        ),
    )

    assert get_current_clip_ratios(config, global_step=1) == pytest.approx((0.2, 0.335))
    assert get_current_clip_ratios(config, global_step=5) == pytest.approx((0.2, 0.275))
    assert get_current_clip_ratios(config, global_step=10) == pytest.approx((0.2, 0.2))


@pytest.mark.parametrize(
    "schedule, match",
    [
        (ClipRatioScheduleConfig(enable=True, type="cosine"), "Unsupported clip_ratio_schedule.type"),
        (ClipRatioScheduleConfig(enable=True, schedule_steps=-1), "schedule_steps"),
        (ClipRatioScheduleConfig(enable=True, clip_high_start=-0.1, schedule_steps=10), "clip_high_start"),
    ],
)
def test_get_current_clip_ratios_rejects_invalid_schedule(schedule, match):
    config = _actor_config(clip_ratio_schedule=schedule)

    with pytest.raises(ValueError, match=match):
        get_current_clip_ratios(config, global_step=0)


def _ppo_loss_inputs(runtime_clip_ratio_low=None, runtime_clip_ratio_high=None):
    old_log_probs = torch.zeros(1, 1)
    advantages = torch.ones(1, 1)
    response_mask = torch.ones(1, 1)
    data = tu.get_tensordict(
        {
            "prompts": torch.tensor([[11]]),
            "responses": torch.tensor([[12]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        },
        non_tensor_dict={"dp_size": 1, "batch_num_tokens": None, "global_batch_size": None},
    )
    if runtime_clip_ratio_low is not None:
        tu.assign_non_tensor(data, clip_ratio_low=runtime_clip_ratio_low)
    if runtime_clip_ratio_high is not None:
        tu.assign_non_tensor(data, clip_ratio_high=runtime_clip_ratio_high)

    model_output = {"log_probs": torch.tensor([math.log(1.3), 0.0])}
    return model_output, data


def test_ppo_loss_uses_runtime_clip_metadata_and_reports_metrics():
    config = _actor_config()
    model_output, data = _ppo_loss_inputs(runtime_clip_ratio_low=0.1, runtime_clip_ratio_high=0.5)

    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data)

    assert loss.item() == pytest.approx(-1.3)
    assert metrics["actor/clip_ratio_low"].aggregate() == pytest.approx(0.1)
    assert metrics["actor/clip_ratio_high"].aggregate() == pytest.approx(0.5)
    assert metrics["actor/clip_bound_lower"].aggregate() == pytest.approx(0.9)
    assert metrics["actor/clip_bound_upper"].aggregate() == pytest.approx(1.5)
    assert config.clip_ratio_low == pytest.approx(0.2)
    assert config.clip_ratio_high == pytest.approx(0.2)


def test_ppo_loss_falls_back_to_static_clip_config_without_metadata():
    config = _actor_config()
    model_output, data = _ppo_loss_inputs()

    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data)

    assert loss.item() == pytest.approx(-1.2)
    assert metrics["actor/clip_ratio_low"].aggregate() == pytest.approx(0.2)
    assert metrics["actor/clip_ratio_high"].aggregate() == pytest.approx(0.2)
