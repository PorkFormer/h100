"""Experimental DAPO trainer hooks for direct Readiness Dominance."""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.experimental.probe_credit.dapo_trainer import (
    RayDAPOProbeCreditTrainer,
    _config_get,
)
from verl.experimental.probe_credit.probe_runtime import (
    ProbeAggregation,
    ProbeTrajectory,
    aggregate_probe_results,
    build_absolute_probe_requests,
    generate_grouped_probe_results,
)
from verl.experimental.probe_credit.readiness_dominance import (
    apply_frontier_reweighting,
    compute_readiness_dominance,
)
from verl.trainer.config import ReadinessDominanceConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer


class RayDAPOReadinessDominanceTrainer(RayDAPOProbeCreditTrainer):
    """Reuse the verified DAPO fit loop with independent dominance hooks."""

    def _dominance_config(self) -> ReadinessDominanceConfig:
        cached = getattr(self, "_typed_readiness_dominance_config", None)
        if cached is None:
            raw_config = self.config.algorithm.readiness_dominance
            cached = (
                raw_config
                if isinstance(raw_config, ReadinessDominanceConfig)
                else omega_conf_to_dataclass(raw_config, ReadinessDominanceConfig)
            )
            self._typed_readiness_dominance_config = cached
        return cached

    def _validate_probe_credit_mode(self) -> None:
        """Validate the independent dominance protocol before inherited fit starts."""
        dominance = self._dominance_config()
        dominance.validate()
        adv_estimator = self.config.algorithm.adv_estimator
        if adv_estimator not in (
            "grpo",
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_VECTORIZED,
        ):
            raise ValueError("Readiness Dominance supports synchronous GRPO only")
        rollout = self.config.actor_rollout_ref.rollout
        if rollout.name != "vllm":
            raise ValueError("Readiness Dominance supports the vLLM rollout backend only")
        probe_credit = _config_get(self.config.algorithm, "probe_credit")
        if probe_credit is not None and bool(_config_get(probe_credit, "enable", False)):
            raise ValueError("Readiness Dominance cannot be combined with ProbeCredit")
        if _config_get(self.config.algorithm, "use_kl_in_reward", False):
            raise ValueError("Readiness Dominance requires algorithm.use_kl_in_reward=false")
        if getattr(self, "use_critic", False):
            raise ValueError("Readiness Dominance supports GRPO with no critic")
        if _config_get(_config_get(rollout, "multi_turn"), "enable", False):
            raise ValueError("Readiness Dominance supports single-turn rollout only")
        if _config_get(_config_get(self.config, "distillation"), "enabled", False):
            raise ValueError("Readiness Dominance does not support distillation")
        if getattr(self, "use_teacher_policy", False):
            raise ValueError("Readiness Dominance does not support a teacher policy")
        rollout_correction = _config_get(self.config.algorithm, "rollout_correction")
        if rollout_correction is not None and any(
            (
                _config_get(rollout_correction, "rollout_is") is not None,
                _config_get(rollout_correction, "rollout_rs") is not None,
                bool(_config_get(rollout_correction, "bypass_mode", False)),
            )
        ):
            raise ValueError("Readiness Dominance does not support rollout correction")
        profiler_steps = _config_get(_config_get(self.config, "global_profiler"), "steps")
        if profiler_steps:
            raise ValueError("Readiness Dominance does not support configured profiling steps")
        filter_groups = _config_get(self.config.algorithm, "filter_groups")
        if not bool(_config_get(filter_groups, "enable", False)):
            raise ValueError("Readiness Dominance requires filter_groups.enable=true")
        if _config_get(filter_groups, "metric") != "acc":
            raise ValueError("Readiness Dominance requires filter_groups.metric=acc")
        actor = _config_get(self.config.actor_rollout_ref, "actor")
        if _config_get(actor, "loss_agg_mode") != "token-mean":
            raise ValueError("Readiness Dominance supports actor loss_agg_mode=token-mean only")
        response_length = _config_get(rollout, "response_length")
        if (
            not isinstance(response_length, int)
            or isinstance(response_length, bool)
            or response_length <= 0
        ):
            raise ValueError("rollout.response_length must be a positive integer")
        if max(dominance.absolute_horizons) >= response_length:
            raise ValueError(
                "every readiness_dominance absolute horizon must be less than "
                f"rollout.response_length={response_length}"
            )

    def _terminal_success_from_acc(
        self, batch: DataProto, metrics: dict[str, float]
    ) -> torch.Tensor:
        """Read authoritative terminal correctness and log score-sum disagreement."""
        values = batch.non_tensor_batch.get("acc")
        if values is None:
            raise ValueError("retained batch is missing authoritative acc")
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or len(array) != len(batch):
            raise ValueError("authoritative acc length must match retained batch")
        normalized: list[bool] = []
        for value in array.tolist():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"authoritative acc must be numeric, got {value!r}") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"authoritative acc must be finite, got {value!r}")
            if numeric not in (0.0, 1.0):
                raise ValueError(f"authoritative acc must be binary, got {value!r}")
            normalized.append(bool(numeric))
        token_scores = batch.batch.get("token_level_scores")
        if token_scores is None or token_scores.ndim != 2 or token_scores.shape[0] != len(batch):
            raise ValueError("token_level_scores must match retained batch for disagreement metrics")
        terminal_success = torch.tensor(
            normalized, dtype=torch.bool, device=token_scores.device
        )
        score_sum_success = token_scores.sum(dim=-1) > 0
        disagreement = terminal_success != score_sum_success
        metrics["dominance/terminal_success_score_disagreement_rate"] = (
            float(disagreement.float().mean().item()) if len(batch) else 0.0
        )
        return terminal_success

    def _prepare_final_retained_batch(
        self, batch: DataProto, metrics: dict[str, float], timing_raw: dict[str, float]
    ) -> DataProto:
        """Select the dominance mode without reading ProbeCreditConfig.enable."""
        self._validate_rollout_policy_version(batch)
        if self._dominance_config().mode != "off":
            terminal_success = self._terminal_success_from_acc(batch, metrics)
            batch = self._probe_final_retained_batch(
                batch, terminal_success, metrics, timing_raw
            )
        self.checkpoint_manager.sleep_replicas()
        return batch

    def _probe_final_retained_batch(
        self,
        batch: DataProto,
        terminal_success: torch.Tensor,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
    ) -> DataProto:
        """Generate absolute-horizon Probes for retained acc-success rows only."""
        config = self._dominance_config()
        rollout_policy_version = self._validate_rollout_policy_version(batch)
        response_mask = batch.batch["response_mask"]
        prompt_width = batch.batch["prompts"].shape[-1]
        prompt_mask = batch.batch["attention_mask"][:, :prompt_width].bool()
        trajectories: list[ProbeTrajectory] = []
        for row in range(len(batch)):
            prompt_ids = tuple(batch.batch["prompts"][row][prompt_mask[row]].tolist())
            response_ids = tuple(
                batch.batch["responses"][row][response_mask[row].bool()].tolist()
            )
            trajectories.append(
                ProbeTrajectory(
                    uid=str(batch.non_tensor_batch["uid"][row]),
                    trajectory_id=str(batch.non_tensor_batch["trajectory_id"][row]),
                    prompt_token_ids=prompt_ids,
                    response_token_ids=response_ids,
                )
            )

        encoded_prefix = self.tokenizer(
            config.answer_prefix, add_special_tokens=False, return_attention_mask=False
        )
        prefix_ids = encoded_prefix["input_ids"]
        rollout = self.config.actor_rollout_ref.rollout
        max_model_len = rollout.max_model_len or rollout.prompt_length + rollout.response_length
        plan = build_absolute_probe_requests(
            trajectories,
            trajectory_mask=terminal_success.tolist(),
            policy_version=rollout_policy_version,
            absolute_horizons=config.absolute_horizons,
            answer_prefix_token_ids=prefix_ids,
            n=config.n,
            max_tokens=config.max_tokens,
            max_model_len=max_model_len,
            strict=config.strict,
        )
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "stop": list(config.stop),
        }
        if plan.requests:
            with marked_timer(
                "dominance_probe_generation_scoring", timing_raw, color="magenta"
            ):
                results = generate_grouped_probe_results(
                    self.llm_server_manager.get_client(),
                    plan.requests,
                    sampling_params=sampling_params,
                    score_candidate=lambda request, text: self._score_probe_candidate(
                        batch, request, text
                    ),
                    max_concurrent_requests=config.max_concurrent_requests,
                    request_batch_size=config.request_batch_size,
                )
            aggregate = aggregate_probe_results(
                plan.requests,
                results,
                trajectory_count=len(batch),
                position_count=len(plan.absolute_horizons),
                n=config.n,
                strict=config.strict,
                expected_policy_version=rollout_policy_version,
            )
        else:
            results = []
            aggregate = ProbeAggregation(
                values=tuple(
                    (0.0,) * len(plan.absolute_horizons) for _ in range(len(batch))
                ),
                valid_mask=plan.valid_mask,
            )
        if config.strict and aggregate.valid_mask != plan.valid_mask:
            raise ValueError("absolute Probe aggregation validity does not match its plan")
        device = batch.batch["responses"].device
        batch.batch["dominance_probe_values"] = torch.tensor(
            aggregate.values, dtype=torch.float32, device=device
        )
        batch.batch["dominance_probe_valid_mask"] = torch.tensor(
            aggregate.valid_mask, dtype=torch.bool, device=device
        )
        batch.batch["dominance_absolute_horizons"] = torch.tensor(
            plan.absolute_horizons, dtype=torch.long, device=device
        ).repeat(len(batch), 1)
        batch.batch["dominance_terminal_success"] = terminal_success.to(device=device)
        output_token_counts = [result.output_token_count for result in results]
        total_output_tokens = sum(output_token_counts)
        valid_denominator = int(terminal_success.sum().item()) * len(
            plan.absolute_horizons
        )
        valid_cells = sum(sum(row) for row in plan.valid_mask)
        metrics.update(
            {
                "dominance/max_concurrent_requests": float(
                    config.max_concurrent_requests
                ),
                "dominance/request_batch_size": float(config.request_batch_size),
                "dominance/request_count": float(len(plan.requests)),
                "dominance/branch_count": float(len(results)),
                "dominance/input_tokens": float(
                    sum(len(request.input_token_ids) for request in plan.requests)
                ),
                "dominance/output_tokens": float(total_output_tokens),
                "dominance/mean_output_tokens": (
                    float(total_output_tokens / len(output_token_counts))
                    if output_token_counts
                    else 0.0
                ),
                "dominance/max_output_tokens": float(max(output_token_counts, default=0)),
                "dominance/probe_valid_cell_rate": (
                    float(valid_cells / valid_denominator)
                    if valid_denominator
                    else 0.0
                ),
            }
        )
        return batch

    def _compute_probe_credit_advantage(
        self, batch: DataProto, metrics: dict[str, float]
    ) -> DataProto:
        """Run direct dominance after standard GRPO; shadow is bitwise read-only."""
        config = self._dominance_config()
        if config.mode == "off":
            return batch
        advantages_before = batch.batch["advantages"].clone()
        returns_before = batch.batch["returns"].clone()
        scores_before = batch.batch["token_level_scores"].clone()
        rewards_before = batch.batch["token_level_rewards"].clone()
        trajectory_ids_before = batch.non_tensor_batch["trajectory_id"].copy()
        positive_trajectory_mask = (
            (
                batch.batch["advantages"].clamp_min(0)
                * batch.batch["response_mask"].to(batch.batch["advantages"].dtype)
            ).sum(dim=-1)
            > 0
        )
        dominance, dominance_metrics = compute_readiness_dominance(
            batch.batch["dominance_probe_values"],
            batch.batch["dominance_probe_valid_mask"],
            batch.batch["dominance_terminal_success"],
            positive_trajectory_mask,
            batch.non_tensor_batch["uid"],
            n=config.n,
            strict_branch_margin=config.strict_branch_margin,
            min_common_positions=config.min_common_positions,
        )
        batch.batch["dominance_frontier_mask"] = dominance.frontier_mask
        batch.batch["dominance_dominated_mask"] = dominance.dominated_mask
        metrics.update(dominance_metrics)
        if config.mode == "reweight":
            new_advantages, weights, reweight_metrics = apply_frontier_reweighting(
                advantages_before,
                batch.batch["response_mask"],
                batch.non_tensor_batch["uid"],
                dominance,
            )
            batch.batch["terminal_advantages"] = advantages_before
            batch.batch["advantages"] = new_advantages
            batch.batch["returns"] = new_advantages
            batch.batch["dominance_weights"] = weights
            metrics.update(reweight_metrics)
        else:
            if not torch.equal(batch.batch["advantages"], advantages_before):
                raise AssertionError(
                    "Readiness Dominance shadow changed standard GRPO advantages"
                )
            if not torch.equal(batch.batch["returns"], returns_before):
                raise AssertionError(
                    "Readiness Dominance shadow changed standard GRPO returns"
                )
        if not torch.equal(batch.batch["token_level_scores"], scores_before):
            raise AssertionError("Readiness Dominance changed token_level_scores")
        if not torch.equal(batch.batch["token_level_rewards"], rewards_before):
            raise AssertionError("Readiness Dominance changed token_level_rewards")
        if (
            batch.non_tensor_batch["trajectory_id"].tolist()
            != trajectory_ids_before.tolist()
        ):
            raise AssertionError("Readiness Dominance changed retained trajectory ordering")
        return batch
