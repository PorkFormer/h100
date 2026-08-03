"""Dedicated synchronous DAPO trainer for Budgeted Success-Support Floor."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from verl import DataProto
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer, _config_get
from verl.experimental.success_support_floor.batch import (
    build_augmented_actor_batch,
    build_support_batch,
)
from verl.experimental.success_support_floor.cache import (
    CacheExpectations,
    SuccessSupportCache,
    reference_model_fingerprint,
    tokenizer_fingerprints,
)
from verl.experimental.success_support_floor.math import compute_support_floor, update_dual_state
from verl.experimental.success_support_floor.state import (
    SuccessSupportFloorState,
    load_state,
    save_state,
    scientific_config_fingerprint,
)
from verl.trainer.config import SuccessSupportFloorConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator, get_current_clip_ratios
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def support_metrics_from_log_probs(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    reference_seq_logprob: torch.Tensor,
    *,
    alpha: float,
    delta: float,
    lambda_value: float,
) -> dict[str, float]:
    """Compute unambiguous scientific metrics from padded response log probabilities."""
    result = compute_support_floor(
        log_probs,
        reference_seq_logprob,
        alpha=alpha,
        response_mask=response_mask,
    )
    ratios = result.log_ratio.detach().float()
    shortfall = result.shortfall.detach().float()
    constraint = float(shortfall.mean().item())
    residual = constraint - delta
    return {
        "support_floor/log_ratio_mean": float(ratios.mean().item()),
        "support_floor/log_ratio_p10": float(torch.quantile(ratios, 0.1).item()),
        "support_floor/log_ratio_p50": float(torch.quantile(ratios, 0.5).item()),
        "support_floor/log_ratio_p90": float(torch.quantile(ratios, 0.9).item()),
        "support_floor/shortfall_mean": constraint,
        "support_floor/active_fraction": float(result.active_fraction.item()),
        "support_floor/constraint_residual": residual,
        "support_floor/primal_feasible": float(constraint <= delta),
        "support_floor/lambda": float(lambda_value),
        "support_floor/complementarity_proxy": float(lambda_value * max(residual, 0.0)),
    }


class RayDAPOSuccessSupportFloorTrainer(RayDAPOProbeCreditTrainer):
    """Reuse the current DAPO loop while keeping BSSF independent of Probe generation."""

    def _success_support_config(self) -> SuccessSupportFloorConfig:
        cached = getattr(self, "_typed_success_support_config", None)
        if cached is None:
            raw = self.config.algorithm.success_support_floor
            cached = (
                raw
                if isinstance(raw, SuccessSupportFloorConfig)
                else omega_conf_to_dataclass(raw, SuccessSupportFloorConfig)
            )
            self._typed_success_support_config = cached
        return cached

    def _constraint_batch_size(self) -> int:
        config = self._success_support_config()
        if config.constraint_batch_size:
            return int(config.constraint_batch_size)
        actor = self.config.actor_rollout_ref.actor
        return int(actor.ppo_mini_batch_size) * int(self.config.actor_rollout_ref.rollout.n)

    def _load_success_support_cache(self) -> None:
        config = self._success_support_config()
        self._success_support_cache = None
        self._lambda = 0.0
        self._violation_ema = 0.0
        self._support_update_count = 0
        self._last_support_step = -1
        if config.mode == "off":
            return
        if not config.cache_path:
            raise ValueError("success_support_floor.cache_path is required outside off mode")
        tokenizer_fp, template_fp = tokenizer_fingerprints(self.tokenizer)
        self._success_support_cache = SuccessSupportCache.load(
            config.cache_path,
            CacheExpectations(
                reference_budget=config.reference_budget,
                support_threshold=config.support_threshold,
                tokenizer_fingerprint=tokenizer_fp,
                chat_template_fingerprint=template_fp,
                logprob_temperature=float(self.config.actor_rollout_ref.rollout.temperature),
                include_eos=True,
            ),
        )
        validation_path = Path(config.cache_path) / "validation_report.json"
        if not validation_path.is_file():
            raise ValueError("BSSF cache log-probability validation report is missing")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("cache_fingerprint") != self._success_support_cache.fingerprint:
            raise ValueError("BSSF cache validation report fingerprint mismatch")
        if validation.get("passed") is not True:
            raise ValueError("BSSF cache log-probability validation has not passed")
        local_reference_path = getattr(self, "_bssf_reference_model_local_path", None)
        if local_reference_path is not None:
            actual_model_hash = reference_model_fingerprint(local_reference_path)
            if actual_model_hash != self._success_support_cache.manifest["reference_model_hash"]:
                raise ValueError("BSSF cache reference model weight hash mismatch")
        if self._constraint_batch_size() > len(self._success_support_cache.prompts):
            raise ValueError(
                "constraint_batch_size exceeds protected prompts; replacement is not allowed"
            )
        self._lambda = float(config.lambda_init) if config.mode == "dual" else 0.0

    def _validate_probe_credit_mode(self) -> None:
        """Validate the standalone BSSF protocol before the inherited fit starts."""
        config = self._success_support_config()
        config.validate()
        probe = _config_get(self.config.algorithm, "probe_credit")
        if probe is not None and bool(_config_get(probe, "enable", False)):
            raise ValueError("BSSF cannot be combined with ProbeCredit")
        dominance = _config_get(self.config.algorithm, "readiness_dominance")
        if dominance is not None and _config_get(dominance, "mode", "off") != "off":
            raise ValueError("BSSF cannot be combined with ReadinessDominance")
        if config.mode == "off":
            super()._validate_probe_credit_mode()
            self._load_success_support_cache()
            return
        if self.config.algorithm.adv_estimator not in (
            "grpo",
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_VECTORIZED,
        ):
            raise ValueError("BSSF supports synchronous GRPO only")
        rollout = self.config.actor_rollout_ref.rollout
        actor = self.config.actor_rollout_ref.actor
        if rollout.name != "vllm":
            raise ValueError("BSSF supports the vLLM rollout backend only")
        if _config_get(self.config.algorithm, "use_kl_in_reward", False) or bool(
            _config_get(actor, "use_kl_loss", False)
        ):
            raise ValueError("BSSF does not support global KL")
        if getattr(self, "use_reference_policy", False):
            raise ValueError("BSSF must not create an online reference-policy worker")
        if getattr(self, "use_critic", False):
            raise ValueError("BSSF supports GRPO with no critic")
        if _config_get(_config_get(rollout, "multi_turn"), "enable", False):
            raise ValueError("BSSF supports single-turn rollout only")
        if _config_get(_config_get(self.config, "distillation"), "enabled", False):
            raise ValueError("BSSF does not support distillation")
        if getattr(self, "use_teacher_policy", False):
            raise ValueError("BSSF does not support a teacher policy")
        correction = _config_get(self.config.algorithm, "rollout_correction")
        if correction is not None and any(
            (
                _config_get(correction, "rollout_is") is not None,
                _config_get(correction, "rollout_rs") is not None,
                bool(_config_get(correction, "bypass_mode", False)),
            )
        ):
            raise ValueError("BSSF does not support rollout correction")
        if _config_get(_config_get(self.config, "global_profiler"), "steps"):
            raise ValueError("BSSF does not support configured profiling steps")
        filter_groups = _config_get(self.config.algorithm, "filter_groups")
        if not bool(_config_get(filter_groups, "enable", False)):
            raise ValueError("BSSF requires filter_groups.enable=true")
        if _config_get(filter_groups, "metric") != "acc":
            raise ValueError("BSSF requires filter_groups.metric=acc")
        if _config_get(actor, "loss_agg_mode") != "token-mean":
            raise ValueError("BSSF supports actor loss_agg_mode=token-mean only")
        if float(rollout.temperature) <= 0.0:
            raise ValueError("BSSF requires a positive actor log-probability temperature")
        self._load_success_support_cache()

    def _should_update_support(self) -> bool:
        config = self._success_support_config()
        return config.mode != "off" and int(self.global_steps) % config.update_interval == 0

    def _build_support_batch(self) -> DataProto:
        cache = self._success_support_cache
        if cache is None:
            raise RuntimeError("support cache has not been loaded")
        config = self._success_support_config()
        witnesses = cache.sample(
            batch_size=self._constraint_batch_size(),
            seed=config.seed,
            global_step=int(self.global_steps),
            support_update_count=self._support_update_count,
        )
        self._support_update_count += 1
        self._last_support_step = int(self.global_steps)
        prompt_tokens = {row["prompt_key"]: row["prompt_token_ids"] for row in cache.prompts}
        rollout = self.config.actor_rollout_ref.rollout
        data_config = _config_get(self.config, "data")
        prompt_width = int(
            _config_get(rollout, "prompt_length", _config_get(data_config, "max_prompt_length"))
        )
        response_width = int(
            _config_get(rollout, "response_length", _config_get(data_config, "max_response_length"))
        )
        return build_support_batch(
            prompt_tokens_by_key=prompt_tokens,
            witnesses=witnesses,
            prompt_width=prompt_width,
            response_width=response_width,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )

    def _compute_shadow_metrics(self) -> dict[str, float]:
        started = time.perf_counter()
        support_batch = self._build_support_batch()
        batch_td = left_right_2_no_padding(support_batch.to_tensordict())
        tu.assign_non_tensor(
            batch_td,
            temperature=float(self.config.actor_rollout_ref.rollout.temperature),
            calculate_entropy=False,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
        config = self._success_support_config()
        metrics = support_metrics_from_log_probs(
            log_probs,
            support_batch.batch["response_mask"].bool(),
            support_batch.batch["support_ref_seq_logprob"],
            alpha=config.alpha,
            delta=config.delta,
            lambda_value=0.0,
        )
        token_count = int(support_batch.batch["response_mask"].sum().item())
        metrics.update(
            {
                "support_floor/cache_protected_prompt_count": float(
                    self._success_support_cache.manifest["protected_prompt_count"]
                ),
                "support_floor/cache_witness_count": float(
                    self._success_support_cache.manifest["witness_count"]
                ),
                "support_floor/sample_prompt_count": float(len(support_batch)),
                "support_floor/sample_witness_count": float(len(support_batch)),
                "support_floor/sample_response_tokens": float(token_count),
                "support_floor/sample_duplicate_prompt_rate": 0.0,
                "support_floor/update_applied": 0.0,
                "support_floor/update_interval": float(config.update_interval),
                "support_floor/violation_ema": 0.0,
                "perf/support_floor_shadow_forward_seconds": time.perf_counter() - started,
                "perf/support_floor_added_tokens": float(token_count),
            }
        )
        return metrics

    def _update_actor_augmented(self, batch: DataProto, *, rl_batch_size: int) -> DataProto:
        """Update on an augmented batch without changing optimizer mini-batch step count."""
        rollout = self.config.actor_rollout_ref.rollout
        actor = self.config.actor_rollout_ref.actor
        ppo_global_mini_batch = int(actor.ppo_mini_batch_size) * int(rollout.n)
        if rl_batch_size <= 0 or rl_batch_size % ppo_global_mini_batch != 0:
            raise ValueError(
                f"RL batch size {rl_batch_size} must be divisible by PPO mini-batch {ppo_global_mini_batch}"
            )
        num_mini_batch = rl_batch_size // ppo_global_mini_batch
        if len(batch) % num_mini_batch != 0:
            raise ValueError(
                f"augmented batch size {len(batch)} must be divisible by num_mini_batch {num_mini_batch}"
            )
        batch.meta_info["multi_turn"] = False
        batch.meta_info["temperature"] = float(rollout.temperature)
        batch_td = left_right_2_no_padding(batch.to_tensordict())
        # Preserve full response_mask for sequence slicing, but exclude witnesses from
        # the PPO token denominator computed by every training engine.
        batch_td["loss_mask"] = batch_td["ppo_response_mask"]
        calculate_entropy = bool(actor.calculate_entropy or actor.entropy_coeff != 0.0)
        clip_ratio_low, clip_ratio_high = get_current_clip_ratios(actor, self.global_steps)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=False,
            global_batch_size=ppo_global_mini_batch,
            mini_batch_size=None,
            num_mini_batch=num_mini_batch,
            epochs=int(actor.ppo_epochs),
            seed=int(actor.data_loader_seed),
            dataloader_kwargs={"shuffle": bool(actor.shuffle)},
            compute_loss=True,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
        )
        output = self.actor_rollout_wg.update_actor(batch_td)
        actor_metrics = rename_dict(tu.get(output, "metrics"), "actor/")
        actor_metrics["perf/mfu/actor"] = actor_metrics.pop("actor/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_metrics})

    def _update_dual(
        self, actor_output: DataProto, metrics: dict[str, float]
    ) -> None:
        raw_metrics = actor_output.meta_info.get("metrics", {})
        reduced = reduce_metrics(
            {
                key: list(value) if isinstance(value, list) else value
                for key, value in raw_metrics.items()
            }
        )
        key = "actor/support_floor_unweighted_shortfall"
        if key not in reduced:
            raise ValueError(f"active BSSF actor output is missing {key}")
        ppo_epochs = int(self.config.actor_rollout_ref.actor.ppo_epochs)
        if ppo_epochs <= 0:
            raise ValueError("actor.ppo_epochs must be positive")

        def summed_metric(name: str) -> float:
            value = raw_metrics[name]
            if isinstance(value, (list, tuple)):
                return sum(float(item) for item in value)
            return float(value)

        observed = summed_metric(key) / ppo_epochs
        config = self._success_support_config()
        next_state = update_dual_state(
            lambda_value=self._lambda,
            violation_ema=self._violation_ema,
            observed_constraint=observed,
            delta=config.delta,
            dual_lr=config.dual_lr,
            ema_beta=config.dual_ema_beta,
            lambda_max=config.lambda_max,
        )
        self._lambda = next_state.lambda_value
        self._violation_ema = next_state.violation_ema
        residual = observed - config.delta
        metrics.update(
            {
                "support_floor/shortfall_mean": observed,
                "support_floor/constraint_residual": residual,
                "support_floor/primal_feasible": float(observed <= config.delta),
                "support_floor/lambda": self._lambda,
                "support_floor/violation_ema": self._violation_ema,
                "support_floor/complementarity_proxy": self._lambda * max(residual, 0.0),
                "support_floor/update_applied": 1.0,
            }
        )
        for source, destination in (
            ("actor/support_floor_log_ratio_mean", "support_floor/log_ratio_mean"),
            ("actor/support_floor_active_fraction", "support_floor/active_fraction"),
        ):
            if source in raw_metrics:
                metrics[destination] = summed_metric(source) / ppo_epochs
        quantile_weight_key = "actor/support_floor_quantile_weight"
        if quantile_weight_key in raw_metrics:
            quantile_weight = summed_metric(quantile_weight_key)
            if quantile_weight > 0.0:
                for source, destination in (
                    ("actor/support_floor_log_ratio_p10", "support_floor/log_ratio_p10"),
                    ("actor/support_floor_log_ratio_p50", "support_floor/log_ratio_p50"),
                    ("actor/support_floor_log_ratio_p90", "support_floor/log_ratio_p90"),
                ):
                    if source in raw_metrics:
                        metrics[destination] = summed_metric(source) / quantile_weight

    def _state_path(self, global_step: int | None = None) -> str:
        step = int(self.global_steps if global_step is None else global_step)
        return os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
            "success_support_floor",
            "state.json",
        )

    def _resume_state_path(self) -> str:
        if self.config.trainer.resume_mode != "resume_path":
            return self._state_path()
        checkpoint_folder = os.path.abspath(self.config.trainer.resume_from_path)
        return os.path.join(checkpoint_folder, "success_support_floor", "state.json")

    def _save_checkpoint(self):
        super()._save_checkpoint()
        config = self._success_support_config()
        if config.mode == "off":
            return
        cache = self._success_support_cache
        if cache is None:
            raise RuntimeError("cannot save BSSF state without a validated cache")
        save_state(
            self._state_path(),
            SuccessSupportFloorState(
                global_step=int(self.global_steps),
                lambda_value=float(self._lambda),
                violation_ema=float(self._violation_ema),
                support_update_count=int(self._support_update_count),
                last_support_step=int(self._last_support_step),
                cache_fingerprint=cache.fingerprint,
                config_fingerprint=scientific_config_fingerprint(config),
            ),
        )

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        config = self._success_support_config()
        if config.mode == "off" or self.config.trainer.resume_mode == "disable":
            return result
        should_resume = self.config.trainer.resume_mode == "resume_path" or int(self.global_steps) > 0
        if not should_resume:
            return result
        cache = self._success_support_cache
        if cache is None:
            raise RuntimeError("cannot restore BSSF state without a validated cache")
        state = load_state(
            self._resume_state_path(),
            expected_global_step=int(self.global_steps),
            expected_cache_fingerprint=cache.fingerprint,
            expected_config_fingerprint=scientific_config_fingerprint(config),
            lambda_max=config.lambda_max,
        )
        if config.mode == "shadow" and state.lambda_value != 0.0:
            raise ValueError("shadow BSSF checkpoint must have lambda=0")
        self._lambda = state.lambda_value
        self._violation_ema = state.violation_ema
        self._support_update_count = state.support_update_count
        self._last_support_step = state.last_support_step
        return result

    def _compute_advantage_and_actor_update(
        self, batch: DataProto, metrics: dict[str, float], timing_raw: dict[str, float]
    ) -> tuple[DataProto, DataProto]:
        config = self._success_support_config()
        if config.mode != "dual":
            batch, actor_output = super()._compute_advantage_and_actor_update(
                batch, metrics, timing_raw
            )
            if config.mode == "shadow" and self._should_update_support():
                metrics.update(self._compute_shadow_metrics())
            return batch, actor_output

        rollout_n = self.config.actor_rollout_ref.rollout.n
        with marked_timer("adv", timing_raw, color="brown"):
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=rollout_n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                ),
                config=self.config.algorithm,
            )
        if not self._should_update_support():
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)
            metrics["support_floor/update_applied"] = 0.0
            metrics["support_floor/lambda"] = float(self._lambda)
            metrics["support_floor/violation_ema"] = float(self._violation_ema)
            return batch, actor_output

        prepare_started = time.perf_counter()
        support_batch = self._build_support_batch()
        actor_batch = build_augmented_actor_batch(
            batch,
            support_batch,
            lambda_value=self._lambda,
            alpha=config.alpha,
            global_support_batch_size=len(support_batch),
        )
        metrics["perf/support_floor_prepare_seconds"] = time.perf_counter() - prepare_started
        update_started = time.perf_counter()
        with marked_timer("update_actor_with_support", timing_raw, color="red"):
            actor_output = self._update_actor_augmented(
                actor_batch,
                rl_batch_size=len(batch),
            )
        metrics["perf/support_floor_forward_backward_seconds"] = (
            time.perf_counter() - update_started
        )
        support_tokens = int(support_batch.batch["response_mask"].sum().item())
        rl_tokens = int(batch.batch["response_mask"].sum().item())
        metrics.update(
            {
                "support_floor/cache_protected_prompt_count": float(
                    self._success_support_cache.manifest["protected_prompt_count"]
                ),
                "support_floor/cache_witness_count": float(
                    self._success_support_cache.manifest["witness_count"]
                ),
                "support_floor/sample_prompt_count": float(len(support_batch)),
                "support_floor/sample_witness_count": float(len(support_batch)),
                "support_floor/sample_response_tokens": float(support_tokens),
                "support_floor/sample_duplicate_prompt_rate": 0.0,
                "support_floor/update_interval": float(config.update_interval),
                "perf/support_floor_added_tokens": float(support_tokens),
                "perf/support_floor_added_token_ratio": float(
                    support_tokens / rl_tokens if rl_tokens else 0.0
                ),
            }
        )
        self._update_dual(actor_output, metrics)
        return batch, actor_output
