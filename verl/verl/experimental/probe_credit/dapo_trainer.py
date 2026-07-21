"""Experimental synchronous DAPO trainer with on-policy Probe credit redistribution.

Dynamic Sampling is migrated from verl-recipe ``dapo/dapo_ray_trainer.py`` at
``e477deff97b15d067f8b4e71f75b80ae58ad64c4``. Compatibility edits use the
current rollout manager, checkpoint-manager step arguments, reward extraction,
and actor APIs; the positive-std selection and accumulation semantics are unchanged.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.probe_credit.dynamic_sampling import (
    filter_dapo_generation_batch,
    select_complete_prompt_groups,
)
from verl.experimental.probe_credit.probe_credit import (
    apply_probe_credit_redistribution,
    build_probe_token_correction,
    compute_group_relative_probe_advantages,
    compute_probe_pseudo_rewards,
    compute_probe_temporal_returns,
)
from verl.experimental.probe_credit.probe_runtime import (
    ProbeTrajectory,
    aggregate_probe_results,
    build_probe_requests,
    generate_grouped_probe_results,
    relative_horizons,
)
from verl.trainer.config import ProbeCreditConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward, get_custom_reward_fn
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


def _config_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(node, key, default)


def _accumulate_timing(destination: dict[str, float], source: dict[str, Any]) -> None:
    """Add per-generation timing into the current optimizer-step totals."""
    for key, value in source.items():
        destination[key] = destination.get(key, 0.0) + float(value)


class RayDAPOProbeCreditTrainer(RayPPOTrainer):
    """Current-API DAPO loop whose optional Probe runs before every actor update."""

    def _probe_config(self) -> ProbeCreditConfig:
        cached = getattr(self, "_typed_probe_credit_config", None)
        if cached is None:
            raw_config = self.config.algorithm.probe_credit
            cached = (
                raw_config
                if isinstance(raw_config, ProbeCreditConfig)
                else omega_conf_to_dataclass(raw_config, ProbeCreditConfig)
            )
            self._typed_probe_credit_config = cached
        return cached

    def _validate_probe_credit_mode(self) -> None:
        probe = self._probe_config()
        probe.validate()
        adv_estimator = self.config.algorithm.adv_estimator
        if adv_estimator not in ("grpo", AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_VECTORIZED):
            raise ValueError("On-policy Probe credit supports synchronous GRPO only")
        if self.config.actor_rollout_ref.rollout.name != "vllm":
            raise ValueError("On-policy Probe credit supports the vLLM rollout backend only")
        if probe.enable and probe.coef <= 0:
            raise ValueError("Enabled Probe credit requires a positive coefficient")
        if _config_get(self.config.algorithm, "use_kl_in_reward", False):
            raise ValueError("The first DAPO Probe Credit trainer requires algorithm.use_kl_in_reward=false")
        if getattr(self, "use_critic", False):
            raise ValueError("The first DAPO Probe Credit trainer supports GRPO with no critic")
        rollout = self.config.actor_rollout_ref.rollout
        if _config_get(_config_get(rollout, "multi_turn"), "enable", False):
            raise ValueError("The first DAPO Probe Credit trainer supports single-turn rollout only")
        if _config_get(_config_get(self.config, "distillation"), "enabled", False):
            raise ValueError("The first DAPO Probe Credit trainer does not support distillation")
        if getattr(self, "use_teacher_policy", False):
            raise ValueError("The first DAPO Probe Credit trainer does not support a teacher policy")
        rollout_correction = _config_get(self.config.algorithm, "rollout_correction")
        if rollout_correction is not None and any(
            (
                _config_get(rollout_correction, "rollout_is") is not None,
                _config_get(rollout_correction, "rollout_rs") is not None,
                bool(_config_get(rollout_correction, "bypass_mode", False)),
            )
        ):
            raise ValueError("The first DAPO Probe Credit trainer does not support rollout correction")
        profiler_steps = _config_get(_config_get(self.config, "global_profiler"), "steps")
        if profiler_steps:
            raise ValueError("The first DAPO Probe Credit trainer does not support configured profiling steps")

    def _publish_rollout_policy_version(self, version: int) -> None:
        """Publish weights and record the version only after every replica updated."""
        version = int(version)
        self.checkpoint_manager.update_weights(version)
        self._rollout_policy_version = version

    def _capture_actual_rollout_policy_version(self, batch: DataProto) -> int:
        """Copy the version emitted by rollout servers into retained metadata."""
        actual_versions = batch.non_tensor_batch.get("global_steps")
        if actual_versions is None:
            raise ValueError("normal rollout is missing actual server global_steps")
        batch.non_tensor_batch["rollout_policy_version"] = np.asarray(actual_versions, dtype=object).copy()
        return self._validate_rollout_policy_version(batch)

    def _validate_rollout_policy_version(self, batch: DataProto) -> int:
        versions = batch.non_tensor_batch.get("rollout_policy_version")
        if versions is None or len(versions) != len(batch):
            raise ValueError("retained batch is missing actual rollout policy version")
        normalized: list[int] = []
        for version in np.asarray(versions, dtype=object).tolist():
            if version is None:
                raise ValueError("retained batch is missing actual rollout policy version")
            try:
                normalized.append(int(version))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid actual rollout policy version: {version!r}") from exc
        unique_versions = set(normalized)
        if len(unique_versions) != 1:
            raise ValueError(f"mixed rollout policy versions: {sorted(unique_versions)}")
        actual_version = next(iter(unique_versions))
        expected_version = getattr(self, "_rollout_policy_version", None)
        if expected_version is None:
            raise ValueError("rollout policy version has not been published")
        if actual_version != int(expected_version):
            raise ValueError(
                f"actual rollout policy version {actual_version} does not match published version {expected_version}"
            )
        return actual_version

    def _prepare_final_retained_batch(
        self, batch: DataProto, metrics: dict[str, float], timing_raw: dict[str, float]
    ) -> DataProto:
        self._validate_rollout_policy_version(batch)
        if self._probe_config().enable:
            batch = self._probe_final_retained_batch(batch, metrics, timing_raw)
        self.checkpoint_manager.sleep_replicas()
        return batch

    def _probe_final_retained_batch(
        self, batch: DataProto, metrics: dict[str, float], timing_raw: dict[str, float]
    ) -> DataProto:
        probe = self._probe_config()
        rollout_policy_version = self._validate_rollout_policy_version(batch)
        positions = tuple(float(position) for position in probe.relative_positions)
        response_mask = batch.batch["response_mask"]
        prompt_width = batch.batch["prompts"].shape[-1]
        prompt_mask = batch.batch["attention_mask"][:, :prompt_width].bool()
        trajectories: list[ProbeTrajectory] = []
        for row in range(len(batch)):
            prompt_ids = tuple(batch.batch["prompts"][row][prompt_mask[row]].tolist())
            response_ids = tuple(batch.batch["responses"][row][response_mask[row].bool()].tolist())
            trajectories.append(
                ProbeTrajectory(
                    uid=str(batch.non_tensor_batch["uid"][row]),
                    trajectory_id=str(batch.non_tensor_batch["trajectory_id"][row]),
                    prompt_token_ids=prompt_ids,
                    response_token_ids=response_ids,
                )
            )

        encoded_prefix = self.tokenizer(probe.answer_prefix, add_special_tokens=False, return_attention_mask=False)
        prefix_ids = encoded_prefix["input_ids"]
        rollout = self.config.actor_rollout_ref.rollout
        max_model_len = rollout.max_model_len or rollout.prompt_length + rollout.response_length
        requests = build_probe_requests(
            trajectories,
            policy_version=rollout_policy_version,
            relative_positions=positions,
            answer_prefix_token_ids=prefix_ids,
            n=probe.n,
            max_tokens=probe.max_tokens,
            max_model_len=max_model_len,
            probe_zero_position=probe.probe_zero_position,
            strict=probe.strict,
        )
        sampling_params = {
            "temperature": probe.temperature,
            "top_p": probe.top_p,
            "top_k": probe.top_k,
            "max_tokens": probe.max_tokens,
            "stop": list(probe.stop),
        }
        with marked_timer("probe_generation_scoring", timing_raw, color="magenta"):
            results = generate_grouped_probe_results(
                self.llm_server_manager.get_client(),
                requests,
                sampling_params=sampling_params,
                score_candidate=lambda request, text: self._score_probe_candidate(batch, request, text),
                max_concurrent_requests=probe.max_concurrent_requests,
                request_batch_size=probe.request_batch_size,
            )
        aggregate = aggregate_probe_results(
            requests,
            results,
            trajectory_count=len(batch),
            position_count=len(positions),
            n=probe.n,
            strict=probe.strict,
            expected_policy_version=rollout_policy_version,
        )
        device = batch.batch["responses"].device
        batch.batch["probe_values"] = torch.tensor(aggregate.values, dtype=torch.float32, device=device)
        batch.batch["probe_valid_mask"] = torch.tensor(aggregate.valid_mask, dtype=torch.bool, device=device)
        batch.batch["probe_relative_positions"] = torch.tensor(positions, dtype=torch.float32, device=device).repeat(
            len(batch), 1
        )
        batch.batch["probe_absolute_horizons"] = torch.tensor(
            [relative_horizons(len(trajectory.response_token_ids), positions) for trajectory in trajectories],
            dtype=torch.long,
            device=device,
        )
        output_token_counts = [result.output_token_count for result in results]
        total_output_tokens = sum(output_token_counts)
        metrics.update(
            {
                "probe_credit/max_concurrent_requests": float(probe.max_concurrent_requests),
                "probe_credit/request_batch_size": float(probe.request_batch_size),
                "probe_credit/request_count": float(len(requests)),
                "probe_credit/branch_count": float(len(results)),
                "probe_credit/input_tokens": float(sum(len(request.input_token_ids) for request in requests)),
                "probe_credit/output_tokens": float(total_output_tokens),
                "probe_credit/mean_output_tokens": (
                    float(total_output_tokens / len(output_token_counts)) if output_token_counts else 0.0
                ),
                "probe_credit/max_output_tokens": float(max(output_token_counts, default=0)),
            }
        )
        for position_index, position in enumerate(positions):
            metrics[f"probe_credit/value_q{position:g}"] = float(
                batch.batch["probe_values"][:, position_index].mean().item()
            )
        return batch

    def _score_probe_candidate(self, batch: DataProto, request: Any, verifier_text: str) -> bool:
        row = request.targets[0].trajectory_index
        data_source = batch.non_tensor_batch["data_source"][row]
        reward_model = batch.non_tensor_batch["reward_model"][row]
        ground_truth = reward_model["ground_truth"]
        extra_info = batch.non_tensor_batch.get("extra_info", np.asarray([{}] * len(batch), dtype=object))[row]
        if not hasattr(self, "_probe_reward_fn"):
            from verl.utils.reward_score import get_default_compute_score

            self._probe_reward_fn = get_custom_reward_fn(self.config) or get_default_compute_score(
                self.config.reward.reward_manager.name
            )
        score = self._probe_reward_fn(
            data_source=data_source,
            solution_str=verifier_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if inspect.isawaitable(score):
            raise TypeError("The first Probe credit implementation requires a synchronous math verifier")
        if isinstance(score, dict):
            score = score.get("score", score.get("reward"))
        return float(score) > 0.0

    def _compute_probe_credit_advantage(
        self, batch: DataProto, metrics: dict[str, float]
    ) -> DataProto:
        probe = self._probe_config()
        if not probe.enable:
            return batch
        probe_values = batch.batch.get("probe_values")
        valid_mask = batch.batch.get("probe_valid_mask")
        if probe_values is None or valid_mask is None:
            raise ValueError("Probe values and valid mask are required before advantage computation")
        if probe_values.shape != valid_mask.shape:
            raise ValueError("probe_valid_mask and probe_values must have the same shape")
        if probe_values.ndim != 2 or probe_values.shape[-1] != 5:
            raise ValueError("Probe values must contain exactly 5 positions")
        if valid_mask.dtype is not torch.bool or not bool(valid_mask.all().item()):
            raise ValueError("all Probe values must be valid before advantage computation")
        if not bool(torch.isfinite(probe_values).all().item()):
            raise ValueError("Probe values must be finite")
        if not bool(((probe_values >= 0.0) & (probe_values <= 1.0)).all().item()):
            raise ValueError("Probe values must be in [0, 1]")
        rewards_before = batch.batch["token_level_rewards"].clone()
        scores_before = batch.batch["token_level_scores"].clone()
        pseudo_rewards = compute_probe_pseudo_rewards(probe_values)
        temporal_returns = compute_probe_temporal_returns(pseudo_rewards, probe.rho)
        probe_advantages, norm_metrics = compute_group_relative_probe_advantages(
            temporal_returns,
            batch.non_tensor_batch["uid"],
            norm_by_std=probe.norm_probe_by_std,
            epsilon=probe.epsilon,
        )
        correction, correction_metrics = build_probe_token_correction(
            probe_advantages, batch.batch["response_mask"]
        )
        if correction_metrics["probe_credit/zero_mass_residual_max"] > 1.0e-5:
            raise ValueError("Probe correction violates raw masked advantage-mass conservation")
        apply_metrics = apply_probe_credit_redistribution(batch, correction, coef=probe.coef, enable=True)
        if not torch.equal(rewards_before, batch.batch["token_level_rewards"]):
            raise AssertionError("Probe credit changed token_level_rewards")
        if not torch.equal(scores_before, batch.batch["token_level_scores"]):
            raise AssertionError("Probe credit changed token_level_scores")
        batch.batch["probe_pseudo_rewards"] = pseudo_rewards
        batch.batch["probe_temporal_returns"] = temporal_returns
        batch.batch["probe_segment_advantages"] = probe_advantages
        batch.batch["probe_correction"] = correction
        metrics.update(norm_metrics)
        metrics.update(correction_metrics)
        metrics.update(apply_metrics)
        return batch

    def _compute_old_and_reference(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch.pop("entropys")
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=batch.batch["response_mask"],
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            metrics.update({"actor/entropy": entropy_agg.detach().item(), "perf/mfu/actor_infer": old_log_prob_mfu})
            batch = batch.union(old_log_prob)
        if self.use_reference_policy:
            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                batch = batch.union(self._compute_ref_log_prob(batch))
        return batch

    def _compute_advantage_and_actor_update(
        self, batch: DataProto, metrics: dict[str, float], timing_raw: dict[str, float]
    ) -> tuple[DataProto, DataProto]:
        """Keep standard GRPO, Probe redistribution, and actor update in explicit order."""
        rollout_n = self.config.actor_rollout_ref.rollout.n
        with marked_timer("adv", timing_raw, color="brown"):
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=rollout_n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )
            batch = self._compute_probe_credit_advantage(batch, metrics)
        with marked_timer("update_actor", timing_raw, color="red"):
            actor_output = self._update_actor(batch)
        return batch, actor_output

    def fit(self):
        """Run official DAPO accumulation, Probe final retained groups, then update."""
        self._validate_probe_credit_mode()
        if self._dump_executor._shutdown:
            self._init_dump_executor()
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        self._publish_rollout_policy_version(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        self.max_steps_duration = 0
        retained_batch: DataProto | None = None
        retained_prompts = 0
        num_gen_batches = 0
        metrics: dict[str, float] = {}
        timing_raw: dict[str, float] = {}
        total_generated_prompt_count = 0
        total_generated_trajectory_count = 0
        total_kept_prompt_count = 0
        total_filtered_prompt_count = 0
        total_generated_response_tokens = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                candidate = DataProto.from_single_dict(batch_dict)
                generated_prompt_count = len(candidate)
                total_generated_prompt_count += generated_prompt_count
                candidate.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                candidate.non_tensor_batch["uid"] = np.asarray(
                    [str(uuid.uuid4()) for _ in range(len(candidate))], dtype=object
                )
                gen_batch = self._get_gen_batch(candidate)
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_input = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
                is_last_step = self.global_steps >= self.total_training_steps
                num_gen_batches += 1

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_output = self.async_rollout_manager.generate_sequences(gen_input)
                        _accumulate_timing(timing_raw, gen_output.meta_info.get("timing", {}))
                        gen_output.meta_info.pop("timing", None)
                    candidate = candidate.repeat(repeat_times=rollout_n, interleave=True).union(gen_output)
                    total_generated_trajectory_count += len(candidate)
                    self._capture_actual_rollout_policy_version(candidate)
                    ordinals: dict[str, int] = {}
                    trajectory_ids = []
                    for uid in candidate.non_tensor_batch["uid"]:
                        uid = str(uid)
                        ordinal = ordinals.get(uid, 0)
                        trajectory_ids.append(f"{uid}:{ordinal}")
                        ordinals[uid] = ordinal + 1
                    candidate.non_tensor_batch["trajectory_id"] = np.asarray(trajectory_ids, dtype=object)
                    candidate.batch["response_mask"] = compute_response_mask(candidate)
                    total_generated_response_tokens += int(candidate.batch["response_mask"].sum().item())

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in candidate.batch:
                            candidate = candidate.union(self._compute_reward_colocate(candidate))
                        reward_tensor, reward_extra_infos = extract_reward(candidate)
                        candidate.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos:
                            candidate.non_tensor_batch.update(
                                {key: np.asarray(value) for key, value in reward_extra_infos.items()}
                            )
                        candidate.batch["token_level_rewards"] = candidate.batch["token_level_scores"]

                    if self.config.algorithm.use_kl_in_reward:
                        candidate = self._compute_old_and_reference(candidate, metrics, timing_raw)
                        candidate, kl_metrics = apply_kl_penalty(
                            candidate, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)

                    if self.config.algorithm.filter_groups.enable:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            candidate.non_tensor_batch[metric_name] = (
                                candidate.batch["token_level_rewards"].sum(-1).cpu().numpy()
                            )
                        elif metric_name == "seq_reward":
                            candidate.non_tensor_batch[metric_name] = (
                                candidate.batch["token_level_scores"].sum(-1).cpu().numpy()
                            )
                        filtered = filter_dapo_generation_batch(candidate, metric_name)
                        kept_count = len(dict.fromkeys(filtered.non_tensor_batch["uid"].tolist()))
                        total_kept_prompt_count += kept_count
                        total_filtered_prompt_count += generated_prompt_count - kept_count
                        retained_prompts += kept_count
                        retained_batch = (
                            filtered if retained_batch is None else DataProto.concat([retained_batch, filtered])
                        )
                        prompt_bsz = self.config.data.train_batch_size
                        if retained_prompts < prompt_bsz:
                            max_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_batches > 0 and num_gen_batches >= max_batches:
                                raise ValueError(
                                    f"num_gen_batches={num_gen_batches} >= max_num_gen_batches={max_batches}. "
                                    "Generated too many; data may be too difficult."
                                )
                            continue
                        batch = select_complete_prompt_groups(retained_batch, prompt_bsz, rollout_n)
                    else:
                        batch = candidate
                        total_kept_prompt_count += generated_prompt_count

                    batch = self._prepare_final_retained_batch(batch, metrics, timing_raw)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self._compute_old_and_reference(batch, metrics, timing_raw)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                        batch, correction_metrics = compute_rollout_correction_and_add_to_batch(
                            batch, rollout_corr_config
                        )
                        metrics.update(correction_metrics)
                    batch, actor_output = self._compute_advantage_and_actor_update(batch, metrics, timing_raw)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    step_rollout_policy_version = self._rollout_policy_version
                    esi_close = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close
                    ):
                        self._save_checkpoint()
                    with marked_timer("update_weights", timing_raw, color="red"):
                        self._publish_rollout_policy_version(self.global_steps)

                metrics["train/num_gen_batches"] = num_gen_batches
                metrics["train/generated_prompt_groups"] = total_generated_prompt_count
                metrics["train/generated_trajectories"] = total_generated_trajectory_count
                metrics["train/retained_prompt_groups"] = len(
                    dict.fromkeys(batch.non_tensor_batch["uid"].tolist())
                )
                metrics["train/filtered_prompt_groups"] = total_filtered_prompt_count
                metrics["train/dynamic_sampling_accept_rate"] = (
                    total_kept_prompt_count / total_generated_prompt_count
                    if total_generated_prompt_count
                    else 0.0
                )
                metrics["train/generated_response_tokens"] = total_generated_response_tokens
                metrics["training/global_step"] = self.global_steps
                metrics["training/rollout_policy_version"] = step_rollout_policy_version
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()
                    )
                )
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
                retained_batch = None
                retained_prompts = 0
                num_gen_batches = 0
                metrics = {}
                timing_raw = {}
                total_generated_prompt_count = 0
                total_generated_trajectory_count = 0
                total_kept_prompt_count = 0
                total_filtered_prompt_count = 0
                total_generated_response_tokens = 0
                self.global_steps += 1
                if is_last_step:
                    self._shutdown_dump_executor()
                    progress_bar.close()
                    return
        self._shutdown_dump_executor()
