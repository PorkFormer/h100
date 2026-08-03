import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

from verl.trainer.ppo.utils import need_critic, need_reference_policy

ROOT = Path(__file__).resolve().parents[3]


def _compose(overrides=None):
    config_dir = ROOT / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="on_policy_budgeted_capability_floor_dapo_trainer",
            overrides=overrides or [],
        )


def test_config_composes_inert_h4096_without_reference_or_critic():
    config = _compose()

    obcf = config.algorithm.on_policy_budgeted_capability_floor
    assert obcf.mode == "off"
    assert obcf.reference_budget == 2048
    assert obcf.base_rollouts_per_prompt == 8
    assert obcf.support_threshold == 2
    assert obcf.reference_tolerance_count == 1
    assert obcf.update_interval == 1
    assert config.actor_rollout_ref.rollout.response_length == 4096
    assert config.actor_rollout_ref.rollout.name == "vllm"
    assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert config.actor_rollout_ref.actor.use_kl_loss is False
    assert config.algorithm.use_kl_in_reward is False
    assert config.algorithm.probe_credit.enable is False
    assert config.algorithm.readiness_dominance.mode == "off"
    assert config.algorithm.success_support_floor.mode == "off"
    assert not need_reference_policy(config)
    assert not need_critic(config)


def test_hydra_overrides_select_shadow_and_dual_modes():
    shadow = _compose(
        [
            "algorithm.on_policy_budgeted_capability_floor.mode=shadow",
            "algorithm.on_policy_budgeted_capability_floor.cache_path=/tmp/cache",
        ]
    )
    dual = _compose(
        [
            "algorithm.on_policy_budgeted_capability_floor.mode=dual",
            "algorithm.on_policy_budgeted_capability_floor.cache_path=/tmp/cache",
            "algorithm.on_policy_budgeted_capability_floor.dual_lr=0.0",
        ]
    )

    assert shadow.algorithm.on_policy_budgeted_capability_floor.mode == "shadow"
    assert dual.algorithm.on_policy_budgeted_capability_floor.mode == "dual"
    assert dual.algorithm.on_policy_budgeted_capability_floor.dual_lr == 0.0


def test_dedicated_entrypoint_selects_obcf_without_worker_paths_for_ref_critic_or_teacher():
    source = (
        ROOT
        / "verl"
        / "experimental"
        / "on_policy_budgeted_capability_floor"
        / "main_dapo_obcf.py"
    ).read_text()

    assert "RayDAPOOnPolicyBudgetedCapabilityFloorTrainer" in source
    assert "_obcf_base_model_local_path" in source
    assert "add_ref_policy_worker" not in source
    assert "add_critic_worker" not in source
    assert "add_teacher_model_resource_pool" not in source
    assert 'config_name="on_policy_budgeted_capability_floor_dapo_trainer"' in source


def test_launchers_have_exact_scales_required_environment_and_no_extra_generation():
    example_dir = ROOT / "examples" / "on_policy_budgeted_capability_floor"
    smoke = example_dir / "train_dapo_qwen3_4b_h4096_obcf_smoke.sh"
    formal = example_dir / "train_dapo_qwen3_4b_h4096_obcf.sh"
    subprocess.run(["bash", "-n", str(smoke)], check=True)
    subprocess.run(["bash", "-n", str(formal)], check=True)
    smoke_text = smoke.read_text()
    formal_text = formal.read_text()

    for text in (smoke_text, formal_text):
        for variable in ("MODEL_PATH", "TRAIN_FILE", "VAL_FILE", "OBCF_CACHE_PATH"):
            assert f'${{{variable}:?' in text
        assert "data.max_response_length=4096" in text
        assert "actor_rollout_ref.rollout.n=8" in text
        assert "algorithm.on_policy_budgeted_capability_floor.reference_budget=2048" in text
        assert "generate_grouped" not in text
        assert "reference_model" not in text
        assert "sbatch" not in text
        assert "srun" not in text

    assert "data.train_batch_size=32" in smoke_text
    assert "trainer.total_training_steps=2" in smoke_text
    assert "algorithm.on_policy_budgeted_capability_floor.mode=shadow" in smoke_text

    assert "data.train_batch_size=256" in formal_text
    assert "trainer.total_training_steps=200" in formal_text
    assert "algorithm.on_policy_budgeted_capability_floor.mode=dual" in formal_text
    assert "algorithm.on_policy_budgeted_capability_floor.update_interval=1" in formal_text
