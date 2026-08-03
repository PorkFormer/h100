# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import copy

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_value_loss,
    get_clip_ratio_metrics,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    runtime_dp_size = float(data["dp_size"])
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    runtime_clip_ratio_low = tu.get_non_tensor_data(data=data, key="clip_ratio_low", default=None)
    runtime_clip_ratio_high = tu.get_non_tensor_data(data=data, key="clip_ratio_high", default=None)
    clip_ratio_low = (
        runtime_clip_ratio_low
        if runtime_clip_ratio_low is not None
        else (config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio)
    )
    clip_ratio_high = (
        runtime_clip_ratio_high
        if runtime_clip_ratio_high is not None
        else (config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio)
    )
    loss_config = config
    if runtime_clip_ratio_low is not None or runtime_clip_ratio_high is not None:
        loss_config = copy.copy(config)
        object.__setattr__(loss_config, "clip_ratio_low", clip_ratio_low)
        object.__setattr__(loss_config, "clip_ratio_high", clip_ratio_high)

    support_fields = {
        "ppo_response_mask",
        "support_response_mask",
        "support_sample_mask",
        "support_ref_seq_logprob",
        "support_log_alpha",
        "support_lambda",
        "support_global_batch_size",
    }
    present_support_fields = support_fields.intersection(data.keys())
    support_enabled = bool(present_support_fields)
    if support_enabled and present_support_fields != support_fields:
        missing = sorted(support_fields - present_support_fields)
        raise ValueError(f"incomplete success-support-floor actor fields: missing {missing}")
    if support_enabled and config.use_kl_loss:
        raise ValueError("success-support-floor actor rows cannot be combined with KL loss")

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    if support_enabled:
        fields.extend(sorted(support_fields))
    data = data.select(*fields).to_padded_tensor()

    response_mask = data.get("ppo_response_mask", data["response_mask"]).to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    if bool(response_mask.any().item()):
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
    else:
        pg_loss = log_prob.sum() * 0.0
        pg_metrics = {}

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics.update(
        Metric.from_dict(get_clip_ratio_metrics(clip_ratio_low, clip_ratio_high), aggregation=AggregationType.MEAN)
    )
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        if bool(response_mask.any().item()):
            entropy_loss = agg_loss(
                loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
            )
        else:
            entropy_loss = entropy.sum() * 0.0
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    if support_enabled:
        support_mask = data["support_response_mask"].to(bool)
        support_rows = data["support_sample_mask"].to(bool)
        if bool((support_mask & response_mask).any().item()):
            raise ValueError("PPO and support response masks must be disjoint")
        if bool((support_mask & ~data["response_mask"].to(bool)).any().item()):
            raise ValueError("support_response_mask must be contained in response_mask")
        if bool((support_mask.any(dim=-1) != support_rows).any().item()):
            raise ValueError("support_sample_mask must identify exactly the support response rows")
        global_sizes = data["support_global_batch_size"].to(torch.long)
        lambdas = data["support_lambda"].float()
        log_alphas = data["support_log_alpha"].float()
        if not bool((global_sizes == global_sizes[0]).all().item()) or int(global_sizes[0].item()) <= 0:
            raise ValueError("support_global_batch_size must be a consistent positive value")
        if not bool((lambdas == lambdas[0]).all().item()) or float(lambdas[0].item()) < 0.0:
            raise ValueError("support_lambda must be a consistent nonnegative value")
        if not bool((log_alphas == log_alphas[0]).all().item()) or not bool(
            torch.isfinite(log_alphas[0]).item()
        ):
            raise ValueError("support_log_alpha must be a consistent finite value")

        differentiable_zero = log_prob.sum() * 0.0
        if bool(support_rows.any().item()):
            current_seq_logprob = (log_prob.float() * support_mask.float()).sum(dim=-1)[support_rows]
            reference_seq_logprob = data["support_ref_seq_logprob"].float()[support_rows]
            if not bool(torch.isfinite(current_seq_logprob).all().item()) or not bool(
                torch.isfinite(reference_seq_logprob).all().item()
            ):
                raise ValueError("success support sequence log probabilities must be finite")
            log_ratio = current_seq_logprob - reference_seq_logprob
            shortfall = torch.relu(log_alphas[0] - log_ratio)
            if not bool(torch.isfinite(shortfall).all().item()):
                raise ValueError("success support shortfall must be finite")
            scale = runtime_dp_size / int(global_sizes[0].item())
            support_unweighted = shortfall.sum() * scale
            support_log_ratio_mean = log_ratio.sum() * scale
            support_active_fraction = (shortfall > 0).float().sum() * scale
            support_quantile_weight = support_rows.sum().float()
            support_quantiles = {
                quantile: torch.quantile(log_ratio.detach(), quantile) * support_quantile_weight
                for quantile in (0.1, 0.5, 0.9)
            }
        else:
            support_unweighted = differentiable_zero
            support_log_ratio_mean = differentiable_zero
            support_active_fraction = differentiable_zero
            support_quantiles = {
                quantile: differentiable_zero.detach() for quantile in (0.1, 0.5, 0.9)
            }
            support_quantile_weight = differentiable_zero.detach()
        support_loss = lambdas[0] * support_unweighted
        policy_loss = policy_loss + support_loss
        metrics["actor/support_floor_loss"] = Metric(
            value=support_loss, aggregation=AggregationType.SUM
        )
        metrics["actor/support_floor_unweighted_shortfall"] = Metric(
            value=support_unweighted, aggregation=AggregationType.SUM
        )
        metrics["actor/support_floor_log_ratio_mean"] = Metric(
            value=support_log_ratio_mean, aggregation=AggregationType.SUM
        )
        metrics["actor/support_floor_active_fraction"] = Metric(
            value=support_active_fraction, aggregation=AggregationType.SUM
        )
        metrics["actor/support_floor_quantile_weight"] = Metric(
            value=support_quantile_weight, aggregation=AggregationType.SUM
        )
        for quantile, value in support_quantiles.items():
            metrics[f"actor/support_floor_log_ratio_p{int(quantile * 100)}"] = Metric(
                value=value, aggregation=AggregationType.SUM
            )

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
