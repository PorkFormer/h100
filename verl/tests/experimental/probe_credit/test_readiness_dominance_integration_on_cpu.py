import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[3]


def test_readiness_dominance_config_composes_off_with_token_mean_dapo():
    config_dir = ROOT / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="readiness_dominance_dapo_trainer")

    assert config.algorithm.adv_estimator == "grpo"
    assert config.algorithm.filter_groups.enable is True
    assert config.algorithm.filter_groups.metric == "acc"
    assert config.algorithm.probe_credit.enable is False
    assert config.algorithm.readiness_dominance.mode == "off"
    assert config.algorithm.readiness_dominance.absolute_horizons == [
        256,
        512,
        1024,
        2048,
    ]
    assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert config.actor_rollout_ref.rollout.name == "vllm"
    assert config.actor_rollout_ref.rollout.response_length == 4096


def test_dedicated_entrypoint_selects_only_readiness_dominance_trainer():
    entrypoint = (
        ROOT
        / "verl"
        / "experimental"
        / "probe_credit"
        / "main_dapo_readiness_dominance.py"
    ).read_text()

    assert "RayDAPOReadinessDominanceTrainer" in entrypoint
    assert "RayDAPOProbeCreditTrainer(" not in entrypoint
    assert "RayPPOTrainer(" not in entrypoint
    assert 'config_name="readiness_dominance_dapo_trainer"' in entrypoint


def test_shadow_smoke_launcher_is_syntax_valid_bounded_and_not_reweightable():
    launcher = (
        ROOT
        / "examples"
        / "readiness_dominance"
        / "train_dapo_qwen3_8b_h100x8_dominance_smoke.sh"
    )

    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert "algorithm.readiness_dominance.mode=shadow" in text
    assert "actor_rollout_ref.rollout.n=2" in text
    assert "algorithm.readiness_dominance.n=2" in text
    assert "algorithm.readiness_dominance.absolute_horizons='[512,1024,2048]'" in text
    assert "algorithm.readiness_dominance.max_tokens=32" in text
    assert "trainer.total_training_steps=1" in text
    assert "trainer.experiment_name=readiness-dominance-shadow-smoke" in text
    assert "trainer.project_name" not in text
    assert "reweight" not in text
    assert "sbatch" not in text
    assert "srun" not in text
