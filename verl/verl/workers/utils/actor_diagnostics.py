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

"""Loss-inert actor diagnostics with one packed DP reduction per optimizer step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import no_padding_2_padding

_BASE_COHORTS = (
    "all",
    "advantage_positive",
    "advantage_negative",
    "position_first",
    "position_middle",
    "position_last",
)
_BOUNDARY_COHORT_FIELDS = (
    "boundary_hit_cap",
    "boundary_eligible",
    "boundary_applied",
    "boundary_changed",
    "boundary_recovered",
    "boundary_regressed",
    "boundary_group_unlocked",
)
_BOUNDARY_REQUIRED_FIELDS = (*_BOUNDARY_COHORT_FIELDS, "boundary_task_delta")
_RATIO_FIELDS = (
    "count",
    "raw_sum",
    "raw_sq_sum",
    "numerical_sum",
    "numerical_sq_sum",
    "effective_sum",
    "effective_sq_sum",
    "nonfinite_count",
    "numerical_clamp_count",
    "clip_lower_count",
    "clip_upper_count",
    "ratio_sum",
    "ratio_sq_sum",
    "entropy_sum",
    "entropy_token_count",
    "entropy_sequence_mean_sum",
    "entropy_trajectory_count",
)
_ENTROPY_BUCKET_FIELDS = (
    "entropy_sum",
    "token_count",
    "sequence_mean_sum",
    "trajectory_count",
)


def _plain_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    keys = ("enable", "numerical_log_ratio_min", "numerical_log_ratio_max", "entropy_bucket_size")
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


def diagnostics_enabled(data: TensorDict) -> bool:
    config = _plain_config(tu.get_non_tensor_data(data, "actor_diagnostics", {}))
    return bool(config.get("enable", False))


@dataclass(frozen=True)
class ActorDiagnosticsResult:
    metrics: dict[str, float]
    reduction_calls: int
    packed_value_count: int


class ActorDiagnosticsAccumulator:
    """Accumulate detached sufficient statistics across loss micro-batches.

    ``finalize`` uses a single SUM all-reduce.  Means, variances, rates and ESS
    are derived after that call, so diagnostics never add a per-micro-batch
    collective.
    """

    def __init__(self, config: Any):
        config = _plain_config(config)
        self.enabled = bool(config.get("enable", False))
        self.log_ratio_min = float(config.get("numerical_log_ratio_min", -20.0))
        self.log_ratio_max = float(config.get("numerical_log_ratio_max", 20.0))
        self.entropy_bucket_size = int(config.get("entropy_bucket_size", 256))
        if not self.log_ratio_min < self.log_ratio_max:
            raise ValueError("actor diagnostics log-ratio bounds must be increasing")
        if self.entropy_bucket_size <= 0:
            raise ValueError("actor diagnostics entropy_bucket_size must be positive")
        self._packed_values: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._has_boundary_labels: bool | None = None
        self.reduction_calls = 0

    @staticmethod
    def _row_field(data: TensorDict, key: str, rows: int, device: torch.device) -> torch.Tensor:
        value = data.get(key)
        if value is None or not torch.is_tensor(value) or value.shape != (rows,):
            raise ValueError(f"actor diagnostics requires row-aligned tensor {key!r}")
        return value.to(device=device, dtype=torch.bool)

    def accumulate(self, model_output: Mapping[str, torch.Tensor], data: TensorDict) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            current = no_padding_2_padding(model_output["log_probs"], data).detach().float()
            selected = data.select("response_mask", "old_log_probs", "advantages").to_padded_tensor()
            response_mask = selected["response_mask"].to(device=current.device, dtype=torch.bool)
            old = selected["old_log_probs"].to(device=current.device, dtype=torch.float32)
            advantages = selected["advantages"].to(device=current.device, dtype=torch.float32)
            if current.shape != old.shape or current.shape != response_mask.shape or current.shape != advantages.shape:
                raise ValueError("actor diagnostics requires aligned current/old/mask/advantage tensors")
            rows, response_width = response_mask.shape
            self._device = current.device

            raw = current - old
            finite = torch.isfinite(raw)
            safe_raw = torch.nan_to_num(
                raw,
                nan=0.0,
                posinf=self.log_ratio_max,
                neginf=self.log_ratio_min,
            )
            numerical = safe_raw.clamp(self.log_ratio_min, self.log_ratio_max)
            ratio = numerical.exp()
            clip_low = float(tu.get_non_tensor_data(data, "clip_ratio_low", 0.2))
            clip_high = float(tu.get_non_tensor_data(data, "clip_ratio_high", 0.2))
            clip_ratio_c = float(tu.get_non_tensor_data(data, "clip_ratio_c", 3.0))
            if clip_ratio_c <= 1.0:
                raise ValueError("actor diagnostics requires dual-clip ratio greater than one")
            lower = 1.0 - clip_low
            upper = 1.0 + clip_high
            positive_effective_ratio = ratio.clamp_max(upper)
            negative_effective_ratio = ratio.clamp_min(lower).clamp_max(clip_ratio_c)
            effective_ratio = torch.where(
                advantages > 0,
                positive_effective_ratio,
                torch.where(advantages < 0, negative_effective_ratio, ratio),
            )
            effective = effective_ratio.log()

            positions = torch.arange(response_width, device=current.device).unsqueeze(0).expand(rows, -1)
            lengths = response_mask.sum(dim=-1).clamp_min(1).unsqueeze(-1)
            first = positions * 3 < lengths
            middle = (positions * 3 >= lengths) & (positions * 3 < lengths * 2)
            last = positions * 3 >= lengths * 2
            cohort_masks: dict[str, torch.Tensor] = {
                "all": response_mask,
                "advantage_positive": response_mask & (advantages > 0),
                "advantage_negative": response_mask & (advantages < 0),
                "position_first": response_mask & first,
                "position_middle": response_mask & middle,
                "position_last": response_mask & last,
            }

            present_boundary = [key in data for key in _BOUNDARY_REQUIRED_FIELDS]
            if any(present_boundary) and not all(present_boundary):
                missing = [
                    key for key, present in zip(_BOUNDARY_REQUIRED_FIELDS, present_boundary, strict=True) if not present
                ]
                raise ValueError(f"incomplete boundary actor diagnostic labels: missing {missing}")
            has_boundary = all(present_boundary)
            if self._has_boundary_labels is None:
                self._has_boundary_labels = has_boundary
            elif self._has_boundary_labels != has_boundary:
                raise ValueError("boundary actor diagnostic labels changed within one optimizer step")
            if has_boundary:
                task_delta = data["boundary_task_delta"]
                if not torch.is_tensor(task_delta) or task_delta.shape != (rows,):
                    raise ValueError("actor diagnostics requires row-aligned tensor 'boundary_task_delta'")
                for key in _BOUNDARY_COHORT_FIELDS:
                    row_mask = self._row_field(data, key, rows, current.device)
                    cohort_masks[key] = response_mask & row_mask.unsqueeze(-1)

            entropy = data.get("old_policy_entropies")
            if entropy is not None:
                if not torch.is_tensor(entropy) or entropy.shape != response_mask.shape:
                    raise ValueError("old_policy_entropies must align with response_mask")
                entropy = entropy.to(device=current.device, dtype=torch.float32).detach()

            # Stack cohort masks and reduce every sufficient-statistic field for
            # all cohorts at once.  The former scalar-at-a-time implementation
            # launched one reduction and one accumulator-add kernel per field
            # and cohort for every PPO micro-batch.  With a per-GPU micro-batch
            # size of one that diagnostic-only launch overhead was measurable.
            # This representation preserves the exact packed key order while
            # reducing the local accumulation to one vector addition.
            masks = torch.stack(tuple(cohort_masks.values()), dim=0)
            valid = masks & finite.unsqueeze(0)
            raw_values = raw.unsqueeze(0)
            numerical_values = numerical.unsqueeze(0)
            effective_values = effective.unsqueeze(0)
            ratio_values = ratio.unsqueeze(0)
            effective_ratio_values = effective_ratio.unsqueeze(0)
            reduce_dims = (1, 2)

            def masked_sum(values: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
                return values.masked_fill(~selection, 0).sum(dim=reduce_dims)

            ratio_fields = [
                valid.sum(dim=reduce_dims),
                masked_sum(raw_values, valid),
                masked_sum(raw_values.square(), valid),
                masked_sum(numerical_values, valid),
                masked_sum(numerical_values.square(), valid),
                masked_sum(effective_values, valid),
                masked_sum(effective_values.square(), valid),
                (masks & ~finite.unsqueeze(0)).sum(dim=reduce_dims),
                (
                    valid
                    & ((raw_values < self.log_ratio_min) | (raw_values > self.log_ratio_max))
                ).sum(dim=reduce_dims),
                (valid & (ratio_values < lower)).sum(dim=reduce_dims),
                (valid & (ratio_values > upper)).sum(dim=reduce_dims),
                masked_sum(effective_ratio_values, valid),
                masked_sum(effective_ratio_values.square(), valid),
            ]

            if entropy is not None:
                entropy_values = entropy.unsqueeze(0)
                entropy_mask = masks & torch.isfinite(entropy_values)
                entropy_row_counts = entropy_mask.sum(dim=-1)
                entropy_row_sums = entropy_values.masked_fill(~entropy_mask, 0).sum(dim=-1)
                entropy_active = entropy_row_counts > 0
                entropy_sequence_means = entropy_row_sums / entropy_row_counts.clamp_min(1)
                ratio_fields.extend(
                    [
                        entropy_row_sums.sum(dim=-1),
                        entropy_row_counts.sum(dim=-1),
                        entropy_sequence_means.masked_fill(~entropy_active, 0).sum(dim=-1),
                        entropy_active.sum(dim=-1),
                    ]
                )
            else:
                zeros = torch.zeros(len(cohort_masks), device=current.device)
                ratio_fields.extend([zeros, zeros, zeros, zeros])

            ratio_packed = torch.stack(
                [field.to(dtype=torch.float64) for field in ratio_fields], dim=-1
            ).reshape(-1)

            if entropy is not None:
                bucket_ids = positions.div(self.entropy_bucket_size, rounding_mode="floor")
                bucket_range = torch.arange(8, device=current.device).view(8, 1, 1)
                bucket_mask = (
                    response_mask.unsqueeze(0)
                    & torch.isfinite(entropy).unsqueeze(0)
                    & (bucket_ids.unsqueeze(0) == bucket_range)
                )
                bucket_row_counts = bucket_mask.sum(dim=-1)
                bucket_row_sums = entropy.unsqueeze(0).masked_fill(~bucket_mask, 0).sum(dim=-1)
                bucket_active = bucket_row_counts > 0
                bucket_sequence_means = bucket_row_sums / bucket_row_counts.clamp_min(1)
                bucket_fields = [
                    bucket_row_sums.sum(dim=-1),
                    bucket_row_counts.sum(dim=-1),
                    bucket_sequence_means.masked_fill(~bucket_active, 0).sum(dim=-1),
                    bucket_active.sum(dim=-1),
                ]
                bucket_packed = torch.stack(
                    [field.to(dtype=torch.float64) for field in bucket_fields], dim=-1
                ).reshape(-1)
            else:
                bucket_packed = torch.zeros(
                    8 * len(_ENTROPY_BUCKET_FIELDS), dtype=torch.float64, device=current.device
                )

            packed = torch.cat((ratio_packed, bucket_packed)).detach()
            if self._packed_values is None:
                self._packed_values = packed
            else:
                self._packed_values.add_(packed)

    def _expected_keys(self) -> list[str]:
        cohorts = list(_BASE_COHORTS)
        if self._has_boundary_labels:
            cohorts.extend(_BOUNDARY_COHORT_FIELDS)
        keys = [f"ratio/{cohort}/{field}" for cohort in cohorts for field in _RATIO_FIELDS]
        keys.extend(f"entropy/bucket_{bucket:02d}/{field}" for bucket in range(8) for field in _ENTROPY_BUCKET_FIELDS)
        return keys

    def finalize(self, dp_group: Any = None) -> ActorDiagnosticsResult:
        if not self.enabled:
            return ActorDiagnosticsResult(metrics={}, reduction_calls=0, packed_value_count=0)
        if self._device is None:
            raise ValueError("enabled actor diagnostics observed no micro-batches")
        keys = self._expected_keys()
        if self._packed_values is None or self._packed_values.numel() != len(keys):
            raise ValueError("actor diagnostics packed sufficient-statistic shape mismatch")
        packed = self._packed_values
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size(dp_group) > 1:
                torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=dp_group)
                self.reduction_calls += 1
        values = {key: float(value) for key, value in zip(keys, packed.cpu().tolist(), strict=True)}
        metrics: dict[str, float] = {
            "actor_diagnostics/dp_reduction_calls": float(self.reduction_calls),
            "actor_diagnostics/packed_value_count": float(len(keys)),
        }
        cohorts = list(_BASE_COHORTS)
        if self._has_boundary_labels:
            cohorts.extend(_BOUNDARY_COHORT_FIELDS)
        for cohort in cohorts:
            prefix = f"ratio/{cohort}"
            count = values[f"{prefix}/count"]
            if count <= 0:
                continue
            raw_mean = values[f"{prefix}/raw_sum"] / count
            numerical_mean = values[f"{prefix}/numerical_sum"] / count
            effective_mean = values[f"{prefix}/effective_sum"] / count
            ratio_sum = values[f"{prefix}/ratio_sum"]
            ratio_sq_sum = values[f"{prefix}/ratio_sq_sum"]
            out = f"actor_diagnostics/{cohort}"
            metrics[f"{out}/token_count"] = count
            metrics[f"{out}/raw_log_ratio_mean"] = raw_mean
            metrics[f"{out}/raw_log_ratio_std"] = max(values[f"{prefix}/raw_sq_sum"] / count - raw_mean**2, 0.0) ** 0.5
            metrics[f"{out}/numerical_log_ratio_mean"] = numerical_mean
            metrics[f"{out}/numerical_log_ratio_std"] = (
                max(values[f"{prefix}/numerical_sq_sum"] / count - numerical_mean**2, 0.0) ** 0.5
            )
            metrics[f"{out}/effective_log_ratio_mean"] = effective_mean
            metrics[f"{out}/effective_log_ratio_std"] = (
                max(values[f"{prefix}/effective_sq_sum"] / count - effective_mean**2, 0.0) ** 0.5
            )
            metrics[f"{out}/nonfinite_fraction"] = values[f"{prefix}/nonfinite_count"] / (
                count + values[f"{prefix}/nonfinite_count"]
            )
            metrics[f"{out}/numerical_clamp_fraction"] = values[f"{prefix}/numerical_clamp_count"] / count
            metrics[f"{out}/clip_lower_exceedance_fraction"] = values[f"{prefix}/clip_lower_count"] / count
            metrics[f"{out}/clip_upper_exceedance_fraction"] = values[f"{prefix}/clip_upper_count"] / count
            metrics[f"{out}/effective_ratio_ess"] = ratio_sum**2 / ratio_sq_sum if ratio_sq_sum > 0 else 0.0
            metrics[f"{out}/effective_ratio_ess_fraction"] = metrics[f"{out}/effective_ratio_ess"] / count
            entropy_count = values[f"{prefix}/entropy_token_count"]
            trajectory_count = values[f"{prefix}/entropy_trajectory_count"]
            if entropy_count > 0:
                metrics[f"{out}/entropy_token_weighted"] = values[f"{prefix}/entropy_sum"] / entropy_count
                metrics[f"{out}/entropy_token_count"] = entropy_count
            if trajectory_count > 0:
                metrics[f"{out}/entropy_sequence_balanced"] = (
                    values[f"{prefix}/entropy_sequence_mean_sum"] / trajectory_count
                )
                metrics[f"{out}/entropy_trajectory_count"] = trajectory_count

        for bucket in range(8):
            prefix = f"entropy/bucket_{bucket:02d}"
            token_count = values[f"{prefix}/token_count"]
            trajectory_count = values[f"{prefix}/trajectory_count"]
            bucket_start = bucket * self.entropy_bucket_size
            bucket_end = (bucket + 1) * self.entropy_bucket_size
            out = f"actor_diagnostics/entropy_bucket_{bucket_start:04d}_{bucket_end:04d}"
            metrics[f"{out}/token_count"] = token_count
            metrics[f"{out}/trajectory_count"] = trajectory_count
            metrics[f"{out}/token_weighted"] = values[f"{prefix}/entropy_sum"] / token_count if token_count else 0.0
            metrics[f"{out}/sequence_balanced"] = (
                values[f"{prefix}/sequence_mean_sum"] / trajectory_count if trajectory_count else 0.0
            )
        return ActorDiagnosticsResult(
            metrics=metrics,
            reduction_calls=self.reduction_calls,
            packed_value_count=len(keys),
        )
