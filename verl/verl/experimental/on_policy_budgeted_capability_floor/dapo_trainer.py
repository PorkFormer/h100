"""Synchronous DAPO trainer for the On-Policy Budgeted Capability Floor."""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from verl import DataProto
from verl.experimental.capability_constraints.identity import (
    reference_model_fingerprint,
    tokenizer_fingerprints,
)
from verl.experimental.capability_constraints.dual import update_projected_dual
from verl.experimental.on_policy_budgeted_capability_floor.cache import (
    CacheExpectations,
    CapabilityFloorCache,
)
from verl.experimental.on_policy_budgeted_capability_floor.math import (
    CapabilityAdvantageResult,
    FloorActionabilityReport,
    compute_capability_advantage,
    summarize_floor_actionability,
)
from verl.experimental.on_policy_budgeted_capability_floor.prefix_batch import (
    ProtectedGroupSelection,
    build_exact_prefix_batch,
    resolve_protected_groups,
)
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    extract_binary_accuracy,
    verifier_pipeline_fingerprint,
)
from verl.experimental.on_policy_budgeted_capability_floor.state import (
    OnPolicyBudgetedCapabilityFloorState,
    load_state,
    save_state,
    scientific_config_fingerprint,
)
from verl.experimental.probe_credit.dapo_trainer import (
    RayDAPOProbeCreditTrainer,
    _config_get,
)
from verl.trainer.config import OnPolicyBudgetedCapabilityFloorConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer


class RayDAPOOnPolicyBudgetedCapabilityFloorTrainer(RayDAPOProbeCreditTrainer):
    """Observe and optionally enforce prompt-level verifier capability floors."""

    def _obcf_config(self) -> OnPolicyBudgetedCapabilityFloorConfig:
        cached = getattr(self, "_typed_obcf_config", None)
        if cached is None:
            raw = self.config.algorithm.on_policy_budgeted_capability_floor
            cached = (
                raw
                if isinstance(raw, OnPolicyBudgetedCapabilityFloorConfig)
                else omega_conf_to_dataclass(raw, OnPolicyBudgetedCapabilityFloorConfig)
            )
            self._typed_obcf_config = cached
        return cached

    def _load_obcf_cache(self) -> None:
        config = self._obcf_config()
        self._obcf_cache = None
        self._obcf_actionability_report = None
        self._lambda = float(config.lambda_init) if config.mode == "dual" else 0.0
        self._violation_ema = 0.0
        self._ema_initialized = False
        self._constraint_observation_count = 0
        self._last_constraint_step = -1
        if config.mode == "off":
            return
        if not config.cache_path:
            raise ValueError("OBCF cache_path is required outside off mode")
        tokenizer_fp, template_fp = tokenizer_fingerprints(self.tokenizer)
        reward = _config_get(self.config, "reward")
        reward_manager = _config_get(reward, "reward_manager")
        reward_manager_module = _config_get(reward_manager, "module")
        custom_reward = _config_get(reward, "custom_reward_function")
        sandbox = _config_get(reward, "sandbox_fusion")
        verifier_fp = verifier_pipeline_fingerprint(
            reward_manager_name=str(_config_get(reward_manager, "name", "naive")),
            reward_manager_source=str(_config_get(reward_manager, "source", "register")),
            reward_manager_module_path=_config_get(reward_manager_module, "path"),
            reward_manager_module_name=_config_get(reward_manager_module, "name"),
            custom_reward_function_path=_config_get(custom_reward, "path"),
            custom_reward_function_name=str(_config_get(custom_reward, "name", "compute_score")),
            custom_reward_kwargs=_config_get(custom_reward, "reward_kwargs", {}),
            reward_kwargs=_config_get(reward, "reward_kwargs", {}),
            sandbox_fusion={
                "url": _config_get(sandbox, "url"),
                "max_concurrent": _config_get(sandbox, "max_concurrent", 64),
                "memory_limit_mb": _config_get(sandbox, "memory_limit_mb", 1024),
            },
        )
        self._obcf_cache = CapabilityFloorCache.load(
            config.cache_path,
            CacheExpectations(
                reference_budget=config.reference_budget,
                base_rollouts_per_prompt=config.base_rollouts_per_prompt,
                support_threshold=config.support_threshold,
                reference_tolerance_count=config.reference_tolerance_count,
                tokenizer_fingerprint=tokenizer_fp,
                chat_template_fingerprint=template_fp,
                verifier_fingerprint=verifier_fp,
            ),
        )
        self._validate_cache_actionability()

    def _validate_cache_actionability(self) -> None:
        """Fail closed for dual caches whose protected floors cannot act on mixed groups."""
        config = self._obcf_config()
        self._obcf_actionability_report: FloorActionabilityReport | None = None
        if config.mode == "off":
            return
        if self._obcf_cache is None:
            raise ValueError("OBCF cache must be loaded before actionability validation")
        report = summarize_floor_actionability(
            cache_rows=self._obcf_cache.prompts,
            current_rollouts_per_prompt=int(self.config.actor_rollout_ref.rollout.n),
        )
        self._obcf_actionability_report = report
        if config.mode == "dual" and report.inert_prompt_count > 0:
            raise ValueError(
                "dual OBCF cache contains structurally inert protected floors: "
                f"{report.inert_prompt_count}/{report.protected_prompt_count}"
            )

    def _cache_actionability_metrics(self) -> dict[str, float]:
        report = getattr(self, "_obcf_actionability_report", None)
        if report is None:
            return {
                "obcf/cache_actionable_prompt_count": 0.0,
                "obcf/cache_inert_prompt_count": 0.0,
                "obcf/cache_inert_prompt_fraction": 0.0,
                "obcf/minimum_positive_empirical_rate": 0.0,
            }
        return {
            "obcf/cache_actionable_prompt_count": float(report.actionable_prompt_count),
            "obcf/cache_inert_prompt_count": float(report.inert_prompt_count),
            "obcf/cache_inert_prompt_fraction": float(report.inert_prompt_fraction),
            "obcf/minimum_positive_empirical_rate": float(
                report.minimum_positive_empirical_rate
            ),
        }

    def _validate_probe_credit_mode(self) -> None:
        config = self._obcf_config()
        config.validate()
        if config.mode == "off":
            super()._validate_probe_credit_mode()
            self._load_obcf_cache()
            return
        probe = _config_get(self.config.algorithm, "probe_credit")
        if bool(_config_get(probe, "enable", False)):
            raise ValueError("OBCF cannot be combined with ProbeCredit")
        dominance = _config_get(self.config.algorithm, "readiness_dominance")
        if _config_get(dominance, "mode", "off") != "off":
            raise ValueError("OBCF cannot be combined with ReadinessDominance")
        witness = _config_get(self.config.algorithm, "success_support_floor")
        if _config_get(witness, "mode", "off") != "off":
            raise ValueError("OBCF cannot be combined with witness BSSF")
        if self.config.algorithm.adv_estimator not in (
            "grpo",
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_VECTORIZED,
        ):
            raise ValueError("OBCF supports synchronous GRPO only")
        rollout = self.config.actor_rollout_ref.rollout
        actor = self.config.actor_rollout_ref.actor
        if rollout.name != "vllm":
            raise ValueError("OBCF supports the vLLM rollout backend only")
        if _config_get(_config_get(rollout, "multi_turn"), "enable", False):
            raise ValueError("OBCF supports single-turn rollout only")
        response_horizon = int(
            _config_get(
                rollout,
                "response_length",
                _config_get(_config_get(self.config, "data"), "max_response_length", 0),
            )
        )
        if config.reference_budget > response_horizon:
            raise ValueError("OBCF reference_budget exceeds the response horizon")
        if getattr(self, "processor", None) is not None:
            raise ValueError("OBCF supports text-only batches")
        reward_model = _config_get(_config_get(self.config, "reward"), "reward_model")
        if bool(_config_get(reward_model, "enable", False)):
            raise ValueError("OBCF does not support a learned reward model verifier")
        model = _config_get(self.config.actor_rollout_ref, "model")
        if _config_get(model, "tokenizer_path") is not None:
            raise ValueError("OBCF does not support a separate verifier tokenizer override")
        if getattr(self, "use_reference_policy", False) or bool(
            _config_get(self.config.algorithm, "use_kl_in_reward", False)
        ) or bool(_config_get(actor, "use_kl_loss", False)):
            raise ValueError("OBCF does not support a reference worker or global KL")
        if getattr(self, "use_critic", False):
            raise ValueError("OBCF supports GRPO with no critic")
        if _config_get(_config_get(self.config, "distillation"), "enabled", False) or getattr(
            self, "use_teacher_policy", False
        ):
            raise ValueError("OBCF does not support distillation or a teacher policy")
        correction = _config_get(self.config.algorithm, "rollout_correction")
        if correction is not None and any(
            (
                _config_get(correction, "rollout_is") is not None,
                _config_get(correction, "rollout_rs") is not None,
                bool(_config_get(correction, "bypass_mode", False)),
            )
        ):
            raise ValueError("OBCF does not support rollout correction")
        if _config_get(_config_get(self.config, "global_profiler"), "steps"):
            raise ValueError("OBCF does not support configured profiling steps")
        filter_groups = _config_get(self.config.algorithm, "filter_groups")
        if not bool(_config_get(filter_groups, "enable", False)) or _config_get(
            filter_groups, "metric"
        ) != "acc":
            raise ValueError("OBCF requires DAPO filtering with metric acc")
        self._load_obcf_cache()

    def _should_observe_constraint(self) -> bool:
        config = self._obcf_config()
        return config.mode != "off"

    def _should_update_dual(self) -> bool:
        config = self._obcf_config()
        return config.mode == "dual" and int(self.global_steps) % config.update_interval == 0

    def _empty_observation_metrics(self) -> dict[str, float]:
        return self._cache_actionability_metrics() | {
            "obcf/protected_prompt_occurrences": 0.0,
            "obcf/protected_rollout_count": 0.0,
            "obcf/prefix_verifier_calls": 0.0,
            "obcf/prefix_token_count": 0.0,
            "obcf/q_current_mean": 0.0,
            "obcf/floor_mean": 0.0,
            "obcf/deficit_mean": 0.0,
            "obcf/active_group_fraction": 0.0,
            "obcf/mixed_group_fraction": 0.0,
            "obcf/all_zero_group_fraction": 0.0,
            "obcf/all_one_group_fraction": 0.0,
            "obcf/nonzero_gradient_group_fraction": 0.0,
            "obcf/capability_advantage_mean": 0.0,
            "obcf/capability_advantage_max_abs": 0.0,
            "obcf/capability_advantage_nonzero_fraction": 0.0,
            "obcf/constraint_residual": 0.0,
            "obcf/primal_feasible": 0.0,
            "obcf/complementarity_proxy": 0.0,
            "obcf/all_zero_active_fraction": 0.0,
            "obcf/active_without_gradient_fraction": 0.0,
            "obcf/update_applied": 0.0,
            "obcf/lambda": float(self._lambda),
            "obcf/violation_ema": float(self._violation_ema),
            "perf/obcf_prepare_seconds": 0.0,
            "perf/obcf_prefix_verifier_seconds": 0.0,
        }

    def _observe_capability(
        self,
        batch: DataProto,
        metrics: dict[str, float],
    ) -> tuple[ProtectedGroupSelection, CapabilityAdvantageResult] | None:
        config = self._obcf_config()
        metrics.update(self._cache_actionability_metrics())
        if not self._should_observe_constraint():
            metrics.update(self._empty_observation_metrics())
            return None
        if any(
            key in batch.non_tensor_batch
            for key in ("multi_modal_data", "multi_modal_inputs")
        ):
            raise ValueError("OBCF supports text-only retained batches")
        started = time.perf_counter()
        selection = resolve_protected_groups(
            batch=batch,
            cache=self._obcf_cache,
            rollout_n=int(self.config.actor_rollout_ref.rollout.n),
        )
        if selection is None:
            metrics.update(self._empty_observation_metrics())
            metrics["perf/obcf_prepare_seconds"] = time.perf_counter() - started
            return None
        prefix_batch = build_exact_prefix_batch(
            batch=batch,
            rollout_indices=selection.rollout_indices,
            reference_budget=config.reference_budget,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        metrics["perf/obcf_prepare_seconds"] = time.perf_counter() - started
        verifier_started = time.perf_counter()
        reward_output = self._score_batch_with_existing_reward_pipeline(prefix_batch)
        prefix_rewards = extract_binary_accuracy(
            reward_output,
            expected_count=len(prefix_batch),
        )
        metrics["perf/obcf_prefix_verifier_seconds"] = time.perf_counter() - verifier_started
        result = compute_capability_advantage(
            prefix_rewards=prefix_rewards,
            group_ids=selection.group_ids,
            capability_floors=selection.capability_floors,
            response_mask=prefix_batch.batch["response_mask"],
            reference_budget=config.reference_budget,
        )
        group_count = len(selection.prompt_keys)
        metrics.update(
            {
                "obcf/protected_prompt_occurrences": float(group_count),
                "obcf/protected_rollout_count": float(len(prefix_batch)),
                "obcf/prefix_verifier_calls": float(len(prefix_batch)),
                "obcf/prefix_token_count": float(prefix_batch.batch["response_mask"].sum().item()),
                "obcf/q_current_mean": float(result.q_current.mean().item()),
                "obcf/floor_mean": float(selection.capability_floors.mean().item()),
                "obcf/deficit_mean": float(result.observed_constraint.item()),
                "obcf/active_group_fraction": float(result.active_group.float().mean().item()),
                "obcf/mixed_group_fraction": float(result.mixed_group_fraction.item()),
                "obcf/all_zero_group_fraction": float(result.all_zero_group_fraction.item()),
                "obcf/all_one_group_fraction": float(result.all_one_group_fraction.item()),
                "obcf/nonzero_gradient_group_fraction": float(
                    result.nonzero_gradient_group_fraction.item()
                ),
                "obcf/lambda": float(self._lambda),
                "obcf/violation_ema": float(self._violation_ema),
                "obcf/update_applied": 0.0,
            }
        )
        return selection, result

    def _compute_advantage_and_actor_update(
        self,
        batch: DataProto,
        metrics: dict[str, float],
        timing_raw: dict[str, float],
    ) -> tuple[DataProto, DataProto]:
        config = self._obcf_config()
        if config.mode == "off":
            return super()._compute_advantage_and_actor_update(batch, metrics, timing_raw)
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
            batch = self._compute_probe_credit_advantage(batch, metrics)
        observation = self._observe_capability(batch, metrics)
        if config.mode == "dual" and observation is not None:
            selection, result = observation
            batch.batch["terminal_advantages"] = batch.batch["advantages"].clone()
            capability_advantages = batch.batch["advantages"].new_zeros(
                batch.batch["advantages"].shape
            )
            capability_advantages[
                selection.rollout_indices, : config.reference_budget
            ] = result.token_advantage
            batch.batch["capability_advantages"] = capability_advantages
            batch.batch["advantages"] = (
                batch.batch["terminal_advantages"]
                + float(self._lambda) * batch.batch["capability_advantages"]
            )
            nonzero = capability_advantages != 0
            metrics.update(
                {
                    "obcf/capability_advantage_mean": float(
                        capability_advantages.mean().item()
                    ),
                    "obcf/capability_advantage_max_abs": float(
                        capability_advantages.abs().max().item()
                    ),
                    "obcf/capability_advantage_nonzero_fraction": float(
                        nonzero.float().mean().item()
                    ),
                }
            )
        with marked_timer("update_actor", timing_raw, color="red"):
            actor_output = self._update_actor(batch)
        if config.mode == "dual" and observation is not None:
            self._update_obcf_dual(
                observation[0],
                observation[1],
                metrics,
                apply_lambda_update=self._should_update_dual(),
            )
        return batch, actor_output

    def _record_constraint_diagnostics(
        self,
        selection: ProtectedGroupSelection,
        result: CapabilityAdvantageResult,
        metrics: dict[str, float],
        *,
        update_applied: bool,
    ) -> None:
        config = self._obcf_config()
        observed = float(result.observed_constraint.item())
        active = result.active_group
        all_zero_active = active & (result.q_current == 0)
        per_rollout_nonzero = result.token_advantage.ne(0).any(dim=1).to(torch.long)
        per_group_nonzero = torch.zeros(
            len(selection.prompt_keys), dtype=torch.long, device=selection.group_ids.device
        )
        per_group_nonzero.scatter_add_(0, selection.group_ids, per_rollout_nonzero)
        active_without_gradient = active & (per_group_nonzero == 0)
        residual = observed - float(config.delta)
        active_count = int(active.sum().item())
        all_zero_active_fraction = (
            float(all_zero_active.sum().item() / active_count) if active_count else 0.0
        )
        active_without_gradient_fraction = (
            float(active_without_gradient.sum().item() / active_count) if active_count else 0.0
        )
        metrics.update(
            {
                "obcf/constraint_residual": residual,
                "obcf/primal_feasible": float(observed <= config.delta),
                "obcf/complementarity_proxy": float(self._lambda * max(residual, 0.0)),
                "obcf/all_zero_active_fraction": all_zero_active_fraction,
                "obcf/active_without_gradient_fraction": active_without_gradient_fraction,
                "obcf/lambda": float(self._lambda),
                "obcf/violation_ema": float(self._violation_ema),
                "obcf/update_applied": float(update_applied),
            }
        )

    def _update_obcf_dual(
        self,
        selection: ProtectedGroupSelection,
        result: CapabilityAdvantageResult,
        metrics: dict[str, float],
        *,
        apply_lambda_update: bool = True,
    ) -> None:
        config = self._obcf_config()
        observed = float(result.observed_constraint.item())
        next_state = update_projected_dual(
            lambda_value=float(self._lambda),
            violation_ema=float(self._violation_ema),
            ema_initialized=bool(self._ema_initialized),
            observed_constraint=observed,
            delta=float(config.delta),
            dual_lr=float(config.dual_lr) if apply_lambda_update else 0.0,
            ema_beta=float(config.dual_ema_beta),
            lambda_max=float(config.lambda_max),
        )
        self._lambda = next_state.lambda_value
        self._violation_ema = next_state.violation_ema
        self._ema_initialized = next_state.ema_initialized
        self._constraint_observation_count += 1
        self._last_constraint_step = int(self.global_steps)
        self._record_constraint_diagnostics(
            selection, result, metrics, update_applied=apply_lambda_update
        )

    def _state_path(self, global_step: int | None = None) -> str:
        step = int(self.global_steps if global_step is None else global_step)
        return os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
            "on_policy_budgeted_capability_floor",
            "state.json",
        )

    def _resume_state_path(self) -> str:
        if self.config.trainer.resume_mode != "resume_path":
            return self._state_path()
        return os.path.join(
            os.path.abspath(self.config.trainer.resume_from_path),
            "on_policy_budgeted_capability_floor",
            "state.json",
        )

    def _save_checkpoint(self):
        super()._save_checkpoint()
        config = self._obcf_config()
        if config.mode == "off":
            return
        if self._obcf_cache is None:
            raise RuntimeError("cannot save OBCF state without a validated cache")
        save_state(
            self._state_path(),
            OnPolicyBudgetedCapabilityFloorState(
                global_step=int(self.global_steps),
                lambda_value=float(self._lambda),
                violation_ema=float(self._violation_ema),
                ema_initialized=bool(self._ema_initialized),
                constraint_observation_count=int(self._constraint_observation_count),
                last_constraint_step=int(self._last_constraint_step),
                cache_fingerprint=self._obcf_cache.fingerprint,
                config_fingerprint=scientific_config_fingerprint(config),
            ),
        )

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        config = self._obcf_config()
        if config.mode != "off" and int(self.global_steps) == 0:
            local_base_path = getattr(self, "_obcf_base_model_local_path", None)
            if local_base_path is None:
                raise ValueError("fresh OBCF runs require a local Base model path for hashing")
            actual_hash = reference_model_fingerprint(local_base_path)
            if actual_hash != self._obcf_cache.manifest["reference_model_hash"]:
                raise ValueError("OBCF cache Base model weight hash mismatch")
        if config.mode == "off" or self.config.trainer.resume_mode == "disable":
            return result
        should_resume = self.config.trainer.resume_mode == "resume_path" or int(self.global_steps) > 0
        if not should_resume:
            return result
        if self._obcf_cache is None:
            raise RuntimeError("cannot restore OBCF state without a validated cache")
        state = load_state(
            self._resume_state_path(),
            expected_global_step=int(self.global_steps),
            expected_cache_fingerprint=self._obcf_cache.fingerprint,
            expected_config_fingerprint=scientific_config_fingerprint(config),
            lambda_max=config.lambda_max,
        )
        if config.mode == "shadow" and (
            state.lambda_value != 0.0
            or state.violation_ema != 0.0
            or state.ema_initialized
            or state.constraint_observation_count != 0
            or state.last_constraint_step != -1
        ):
            raise ValueError("shadow OBCF checkpoint must have unchanged dual state")
        self._lambda = state.lambda_value
        self._violation_ema = state.violation_ema
        self._ema_initialized = state.ema_initialized
        self._constraint_observation_count = state.constraint_observation_count
        self._last_constraint_step = state.last_constraint_step
        return result
