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

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[3]


def test_dedicated_config_composes_default_off_and_disables_other_interventions():
    config_dir = ROOT / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="natural_continuation_boundary_return_dapo_trainer")

    assert config.actor_rollout_ref.rollout.boundary_return.mode == "off"
    assert config.actor_rollout_ref.rollout.boundary_return.long_response_length == 8192
    assert config.algorithm.filter_groups.enable is True
    assert config.algorithm.filter_groups.metric == "acc"
    assert config.algorithm.adv_estimator == "grpo"
    assert config.algorithm.use_kl_in_reward is False
    assert config.algorithm.probe_credit.enable is False
    assert config.algorithm.censor_aware_advantage.enable is False
    assert config.algorithm.readiness_dominance.mode == "off"
    assert config.algorithm.success_support_floor.mode == "off"
    assert config.algorithm.on_policy_budgeted_capability_floor.mode == "off"
    assert config.critic.enable is False


def test_entrypoint_selects_only_boundary_return_trainer():
    source = (
        ROOT
        / "verl"
        / "experimental"
        / "natural_continuation_boundary_return"
        / "main_dapo_boundary_return.py"
    ).read_text(encoding="utf-8")
    assert "RayDAPOBoundaryReturnTrainer(" in source
    assert "RayDAPOProbeCreditTrainer(" not in source
    assert 'config_name="natural_continuation_boundary_return_dapo_trainer"' in source
    preflight_positions = [
        index for index in range(len(source)) if source.startswith("validate_boundary_return_preflight(config)", index)
    ]
    assert len(preflight_positions) == 2
    assert preflight_positions[0] < source.index("self.add_actor_rollout_worker(config)")
    assert preflight_positions[0] < source.index("copy_to_local(")
    assert preflight_positions[0] < source.index("self.init_resource_pool_mgr(config)")
    assert preflight_positions[0] < source.index("trainer.init_workers()")


def test_shadow_documentation_limits_noop_claim_to_identical_normal_candidate_batches():
    text = (ROOT / "docs" / "natural_continuation_boundary_return.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "Given the same normal candidate batches" in normalized
    assert "does not claim bitwise equality of subsequent real-vLLM rollouts" in normalized


def test_formal_example_keeps_h2048_actor_recipe_and_only_expands_continuation_context():
    script = (
        ROOT
        / "examples"
        / "natural_continuation_boundary_return"
        / "run_qwen3_4b_boundary_return_h2048_l8192_replace_fsdp.sh"
    ).read_text(encoding="utf-8")
    required_baseline_fragments = (
        "data.max_prompt_length=1024",
        "data.max_response_length=2048",
        "data.train_batch_size=256",
        "+data.gen_batch_size=768",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.rollout.max_model_len=9216",
        "actor_rollout_ref.rollout.enforce_eager=false",
        "algorithm.filter_groups.metric=acc",
        "actor_rollout_ref.rollout.boundary_return.mode=replace",
        "actor_rollout_ref.rollout.boundary_return.long_response_length=8192",
        "actor_rollout_ref.rollout.boundary_return.correctness_key=acc",
        "actor_rollout_ref.rollout.boundary_return.task_score_key=score",
        "qwen3_4b_base_boundary_return_b256_g768_m16_n8_h2048_l8192_s600_v1",
    )
    for fragment in required_baseline_fragments:
        assert fragment in script
    assert "forced_answer_probe.enable=true" not in script
    assert "censor_aware_advantage.enable=true" not in script
    assert "actor_rollout_ref.rollout.response_length=8192" not in script


def test_documentation_states_replacement_prefix_only_semantics_and_exact_denominators():
    text = (ROOT / "docs" / "natural_continuation_boundary_return.md").read_text(encoding="utf-8")
    for statement in (
        "corrected = original_prefix_shaped + long_task - short_task",
        "replacement, not a bonus",
        "before Dynamic Sampling",
        "prefix-only actor",
        "extra_generated_token_ratio",
        "recovered_rate_given_cap_failure",
        "regressed_rate_given_cap_success",
        "unlocked_group_rate",
        "does not establish efficacy",
    ):
        assert statement in text


def test_generated_reference_configs_include_default_off_boundary_return():
    for name in (
        "_generated_ppo_trainer.yaml",
        "_generated_ppo_megatron_trainer.yaml",
        "_generated_ppo_torchtitan_trainer.yaml",
        "_generated_ppo_veomni_trainer.yaml",
    ):
        text = (ROOT / "verl" / "trainer" / "config" / name).read_text(encoding="utf-8")
        assert "boundary_return:" in text
        assert "long_response_length: 8192" in text
