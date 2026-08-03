"""Synchronous DAPO trainer for the On-Policy Budgeted Capability Floor."""

from __future__ import annotations

import time
from typing import Any

from verl import DataProto
from verl.experimental.capability_constraints.identity import (
    reference_model_fingerprint,
    tokenizer_fingerprints,
)
from verl.experimental.on_policy_budgeted_capability_floor.cache import (
    CacheExpectations,
    CapabilityFloorCache,
)
from verl.experimental.on_policy_budgeted_capability_floor.math import (
    CapabilityAdvantageResult,
    compute_capability_advantage,
)
from verl.experimental.on_policy_budgeted_capability_floor.prefix_batch import (
    ProtectedGroupSelection,
    build_exact_prefix_batch,
    resolve_protected_groups,
)
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    extract_binary_accuracy,
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
        self._obcf_cache = CapabilityFloorCache.load(
            config.cache_path,
            CacheExpectations(
                reference_budget=config.reference_budget,
                base_rollouts_per_prompt=config.base_rollouts_per_prompt,
                support_threshold=config.support_threshold,
                reference_tolerance_count=config.reference_tolerance_count,
                tokenizer_fingerprint=tokenizer_fp,
                chat_template_fingerprint=template_fp,
            ),
        )
        resume_mode = _config_get(_config_get(self.config, "trainer"), "resume_mode", "disable")
        local_base_path = getattr(self, "_obcf_base_model_local_path", None)
        if resume_mode == "disable" and local_base_path is not None:
            actual_hash = reference_model_fingerprint(local_base_path)
            if actual_hash != self._obcf_cache.manifest["reference_model_hash"]:
                raise ValueError("OBCF cache Base model weight hash mismatch")

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
        filter_groups = _config_get(self.config.algorithm, "filter_groups")
        if not bool(_config_get(filter_groups, "enable", False)) or _config_get(
            filter_groups, "metric"
        ) != "acc":
            raise ValueError("OBCF requires DAPO filtering with metric acc")
        self._load_obcf_cache()

    def _should_observe_constraint(self) -> bool:
        config = self._obcf_config()
        return config.mode != "off" and int(self.global_steps) % config.update_interval == 0

    def _observe_capability(
        self,
        batch: DataProto,
        metrics: dict[str, float],
    ) -> tuple[ProtectedGroupSelection, CapabilityAdvantageResult] | None:
        config = self._obcf_config()
        if not self._should_observe_constraint():
            metrics.update(
                {
                    "obcf/update_applied": 0.0,
                    "obcf/lambda": float(self._lambda),
                    "obcf/violation_ema": float(self._violation_ema),
                }
            )
            return None
        if "multi_modal_data" in batch.non_tensor_batch:
            raise ValueError("OBCF supports text-only retained batches")
        started = time.perf_counter()
        selection = resolve_protected_groups(
            batch=batch,
            cache=self._obcf_cache,
            rollout_n=int(self.config.actor_rollout_ref.rollout.n),
        )
        metrics["perf/obcf_prepare_seconds"] = time.perf_counter() - started
        if selection is None:
            metrics.update(
                {
                    "obcf/protected_prompt_occurrences": 0.0,
                    "obcf/protected_rollout_count": 0.0,
                    "obcf/prefix_verifier_calls": 0.0,
                    "obcf/prefix_token_count": 0.0,
                    "obcf/update_applied": 0.0,
                    "obcf/lambda": float(self._lambda),
                    "obcf/violation_ema": float(self._violation_ema),
                    "perf/obcf_prefix_verifier_seconds": 0.0,
                }
            )
            return None
        prefix_batch = build_exact_prefix_batch(
            batch=batch,
            rollout_indices=selection.rollout_indices,
            reference_budget=config.reference_budget,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
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
        self._observe_capability(batch, metrics)
        with marked_timer("update_actor", timing_raw, color="red"):
            actor_output = self._update_actor(batch)
        return batch, actor_output
