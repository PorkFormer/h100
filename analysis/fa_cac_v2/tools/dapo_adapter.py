"""Guarded in-memory FA-CAC v2 adapter for the formal canonical DAPO loop."""

from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import textwrap
from pathlib import Path

from recipe.dapo.main_dapo import DAPOTaskRunner as CanonicalDAPOTaskRunner

CANONICAL_DAPO_REPO = Path("/workspace/rl/verl")
EXPECTED_DAPO_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
EXPECTED_DAPO_FIT_SHA256 = "e75af411cabf44003d4e7f2f7aed16549ac97b023480afa54840ae64517bf63c"
EXPECTED_V1_ADAPTER_SHA256 = "c71b5480c568d4655d057c0106079a486d9dbebb7bfbfb536f9868d031c45120"
EXPECTED_STEP200_V1_FIT_SHA256 = "39b3693224e4fb37e63b478bfbbda23d801de1f5561b7c974aab87db539e0946"
HISTORICAL_V1_ADAPTER = Path("/workspace/rl/h100/analysis/fa_tr_v1/tools/matched_dapo_main.py")
HISTORICAL_EXECUTED_V1_FIT = Path(
    "/workspace/rl/h100/analysis/fa_tr_v1/smoke_s5/provenance/executed_dapo_fit.py"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def attest_canonical_sources() -> dict[str, str]:
    """Attest detached canonical code and immutable historical v1 evidence."""
    commit = subprocess.check_output(
        ["git", "-C", str(CANONICAL_DAPO_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_DAPO_COMMIT:
        raise RuntimeError(f"canonical DAPO commit mismatch: expected={EXPECTED_DAPO_COMMIT} actual={commit}")
    status = subprocess.check_output(
        ["git", "-C", str(CANONICAL_DAPO_REPO), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("canonical DAPO repository is not clean; refusing source patch")
    v1_digest = _sha256_bytes(HISTORICAL_V1_ADAPTER.read_bytes())
    if v1_digest != EXPECTED_V1_ADAPTER_SHA256:
        raise RuntimeError(
            f"historical v1 adapter mismatch: expected={EXPECTED_V1_ADAPTER_SHA256} actual={v1_digest}"
        )
    executed_v1_digest = _sha256_bytes(HISTORICAL_EXECUTED_V1_FIT.read_bytes())
    if executed_v1_digest != EXPECTED_STEP200_V1_FIT_SHA256:
        raise RuntimeError(
            "executed v1 fit mismatch: "
            f"expected={EXPECTED_STEP200_V1_FIT_SHA256} actual={executed_v1_digest}"
        )
    return {
        "canonical_commit": commit,
        "historical_v1_adapter_sha256": v1_digest,
        "executed_v1_fit_sha256": executed_v1_digest,
    }


def build_patched_dapo_fit_source(canonical_source: str) -> str:
    """Build the audited v2 fit method from the exact canonical source text."""
    digest = _sha256_bytes(canonical_source.encode())
    if digest != EXPECTED_DAPO_FIT_SHA256:
        raise RuntimeError(
            "canonical DAPO fit source changed; refusing unaudited execution: "
            f"expected={EXPECTED_DAPO_FIT_SHA256} actual={digest}"
        )
    # The guarded anchors are written at repository-file indentation. Inspect
    # returns a dedented method, so restore one class level while patching.
    source_for_patch = textwrap.indent(canonical_source, "    ")

    validation_anchor = '''        from verl.utils.tracking import Tracking

        logger = Tracking(
'''
    validation_replacement = '''        from verl.utils.tracking import Tracking
        from verl.utils.config import validate_censor_aware_advantage_config

        # Enforce the four-mode truth table before validation, rollout, or updates.
        validate_censor_aware_advantage_config(self.config)

        logger = Tracking(
'''
    if source_for_patch.count(validation_anchor) != 1:
        raise RuntimeError("DAPO validation insertion anchor is not unique")
    source = source_for_patch.replace(validation_anchor, validation_replacement)

    generation_anchor = '''                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
'''
    generation_replacement = '''                num_gen_batches += 1
                metrics["fa_cac/pre_filter_generation_batch_count"] = float(num_gen_batches)
                gen_batch = self._get_gen_batch(new_batch)
'''
    if source.count(generation_anchor) != 1:
        raise RuntimeError("DAPO generation-count insertion anchor is not unique")
    source = source.replace(generation_anchor, generation_replacement)

    probe_anchor = '''                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
'''
    probe_replacement = '''                    # Probe only the final retained PPO trajectories. Oversampled and
                    # group-filtered rows never receive FA evidence.
                    probe_capture = None
                    probe_reward_batch = None
                    if self._forced_answer_probe_enabled():
                        with marked_timer("forced_answer_probe_generation", timing_raw, "magenta"):
                            probe_capture = self._generate_forced_answer_probe_with_replica_cleanup(
                                batch, curr_step_profile=curr_step_profile
                            )
                        batch.non_tensor_batch["__forced_answer_probe_parent_index__"] = np.arange(
                            len(batch), dtype=np.int64
                        )
                        if probe_capture.generations:
                            from verl.trainer.ppo.forced_answer_probe import build_probe_reward_batch

                            probe_reward_batch = build_probe_reward_batch(
                                batch, probe_capture.generations, pad_token_id=self.tokenizer.pad_token_id
                            )
                    else:
                        self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
'''
    if source.count(probe_anchor) != 1:
        raise RuntimeError("DAPO retained-batch probe insertion anchor is not unique")
    source = source.replace(probe_anchor, probe_replacement)

    evidence_anchor = '''                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
'''
    evidence_replacement = '''                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    censor_evidence = None
                    if probe_capture is not None:
                        with marked_timer("forced_answer_probe_reward", timing_raw, "magenta"):
                            original_reward_tensor, original_reward_extra_infos = extract_reward(batch)
                            probe_score = self._score_forced_answer_probe(
                                batch=batch,
                                capture=probe_capture,
                                probe_reward_batch=probe_reward_batch,
                                original_reward_tensor=original_reward_tensor,
                                original_reward_extra_infos=original_reward_extra_infos,
                            )
                            metrics.update(probe_score.diagnostics.metrics)
                            metrics.update(probe_score.training_credit.metrics)
                            censor_evidence = probe_score.censor_evidence
                            # v1 replaces reward before GRPO; CAC keeps original rewards.
                            if self.config.actor_rollout_ref.rollout.forced_answer_probe.training_credit.enable:
                                effective_reward_tensor = probe_score.training_credit.effective_reward_tensor
                                batch.batch["token_level_scores"] = effective_reward_tensor
                                if not self.config.algorithm.use_kl_in_reward:
                                    batch.batch["token_level_rewards"] = effective_reward_tensor

                    if not self.config.algorithm.use_kl_in_reward:
'''
    if source.count(evidence_anchor) != 1:
        raise RuntimeError("DAPO evidence insertion anchor is not unique")
    source = source.replace(evidence_anchor, evidence_replacement)

    advantage_anchor = '''                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
'''
    advantage_replacement = '''                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        from verl.trainer.ppo.censor_aware_advantage import apply_fa_cac_post_advantage_hook

                        batch, fa_cac_metrics = apply_fa_cac_post_advantage_hook(
                            batch, evidence=censor_evidence, algorithm_config=self.config.algorithm
                        )
                        metrics.update(fa_cac_metrics)

                    # update critic
'''
    if source.count(advantage_anchor) != 1:
        raise RuntimeError("DAPO post-advantage hook insertion anchor is not unique")
    return textwrap.dedent(source.replace(advantage_anchor, advantage_replacement))


def patch_dapo_fit() -> str:
    """Attest and patch only the in-memory canonical trainer method."""
    attest_canonical_sources()
    from recipe.dapo import dapo_ray_trainer

    trainer_class = dapo_ray_trainer.RayDAPOTrainer
    canonical_source = textwrap.dedent(inspect.getsource(trainer_class.fit))
    source = build_patched_dapo_fit_source(canonical_source)
    audit_path = os.environ.get("FA_CAC_EXECUTED_FIT_PATH")
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    namespace = dapo_ray_trainer.__dict__
    exec(compile(source, "<fa_cac_v2_matched_dapo_fit>", "exec"), namespace)
    trainer_class.fit = namespace["fit"]
    digest = _sha256_bytes(source.encode())
    trainer_class._fa_cac_v2_matched_fit_sha256 = digest
    return digest


def patch_vllm_grouped_output_transport() -> None:
    """Retain the historical FINAL_ONLY grouped-output compatibility patch."""
    from vllm.sampling_params import RequestOutputKind
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    if getattr(vLLMHttpServer, "_fa_cac_grouped_final_only", False):
        return
    original = vLLMHttpServer.generate_grouped

    async def generate_grouped_final_only(self, *args, **kwargs):
        if "sampling_params" in kwargs:
            params = dict(kwargs["sampling_params"])
            kwargs["sampling_params"] = params
        elif len(args) >= 2:
            args = list(args)
            params = dict(args[1])
            args[1] = params
            args = tuple(args)
        else:
            raise RuntimeError("grouped vLLM request is missing sampling_params")
        params["output_kind"] = RequestOutputKind.FINAL_ONLY
        return await original(self, *args, **kwargs)

    vLLMHttpServer.generate_grouped = generate_grouped_final_only
    vLLMHttpServer._fa_cac_grouped_final_only = True


class MatchedFACACDAPOTaskRunner(CanonicalDAPOTaskRunner):
    """Apply the attested patch inside the remote canonical DAPO task."""

    def run(self, config) -> None:
        digest = patch_dapo_fit()
        patch_vllm_grouped_output_transport()
        print(f"FA-CAC v2 remote matched DAPO fit active: sha256={digest}")
        print("FA-CAC v2 vLLM grouped transport active: output_kind=FINAL_ONLY")
        super().run(config)


def patch_resource_pool_node_affinity() -> None:
    """Constrain the formal run to an explicitly selected physical node."""
    target_ip = os.environ.get("FA_CAC_TARGET_NODE_IP")
    if not target_ip:
        raise RuntimeError("FA_CAC_TARGET_NODE_IP must explicitly select one physical node")
    import ray
    from ray.util.placement_group import placement_group
    from verl.single_controller.ray import base

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=True)
    expected = f"node:{target_ip}"
    if expected not in ray.cluster_resources():
        raise RuntimeError(f"selected Ray node resource is unavailable: {expected}")

    def pinned_get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if self.pgs is not None:
            return self.pgs
        prefix = name or f"{self.name_prefix}verl_group_{'_'.join(map(str, self._store))}:"
        device_name = "NPU" if device_name == "npu" else "GPU" if device_name == "cuda" else device_name
        bundle = {"CPU": self.max_colocate_count, expected: 1e-4}
        if self.use_gpu:
            bundle[device_name] = 1
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        schemes = [[bundle.copy() for _ in range(count)] for count in self._store]
        lifetime = "detached" if self.detached else None
        groups = [
            placement_group(bundles=bundles, strategy=strategy, name=prefix + str(i), lifetime=lifetime)
            for i, bundles in enumerate(schemes)
        ]
        ray.get([group.ready() for group in groups])
        self.pgs = base.sort_placement_group_by_node_ip(groups)
        return self.pgs

    base.RayResourcePool.get_placement_groups = pinned_get_placement_groups
