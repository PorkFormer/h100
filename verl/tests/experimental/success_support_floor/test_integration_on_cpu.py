import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from verl import DataProto
from verl.experimental.success_support_floor.batch import (
    build_augmented_actor_batch,
    build_support_batch,
)
from verl.experimental.success_support_floor.dapo_trainer import (
    RayDAPOSuccessSupportFloorTrainer,
)
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig

ROOT = Path(__file__).resolve().parents[3]


def _compose(overrides=None):
    config_dir = ROOT / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="success_support_floor_dapo_trainer",
            overrides=overrides or [],
        )


def test_config_composes_inert_h4096_without_ref_or_critic():
    config = _compose()
    assert config.algorithm.success_support_floor.mode == "off"
    assert config.algorithm.success_support_floor.reference_budget == 2048
    assert config.actor_rollout_ref.rollout.response_length == 4096
    assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert config.algorithm.probe_credit.enable is False
    assert config.algorithm.readiness_dominance.mode == "off"
    assert not need_reference_policy(config)
    assert not need_critic(config)


def test_hydra_overrides_select_shadow_and_dual_modes():
    shadow = _compose(["algorithm.success_support_floor.mode=shadow"])
    dual = _compose(
        [
            "algorithm.success_support_floor.mode=dual",
            "algorithm.success_support_floor.alpha=0.75",
            "algorithm.success_support_floor.delta=0.0",
        ]
    )
    assert shadow.algorithm.success_support_floor.mode == "shadow"
    assert dual.algorithm.success_support_floor.mode == "dual"
    assert dual.algorithm.success_support_floor.alpha == 0.75
    assert dual.algorithm.success_support_floor.delta == 0.0


def test_dedicated_entrypoint_selects_only_bssf_trainer():
    entrypoint = (
        ROOT
        / "verl"
        / "experimental"
        / "success_support_floor"
        / "main_dapo_success_support_floor.py"
    ).read_text()
    assert "RayDAPOSuccessSupportFloorTrainer" in entrypoint
    assert "RayDAPOProbeCreditTrainer(" not in entrypoint
    assert "RayPPOTrainer(" not in entrypoint
    assert 'config_name="success_support_floor_dapo_trainer"' in entrypoint


def test_bssf_trainer_adds_no_generation_verifier_or_reference_path():
    source = inspect.getsource(
        __import__(
            "verl.experimental.success_support_floor.dapo_trainer",
            fromlist=["unused"],
        )
    )
    assert "generate_sequences" not in source
    assert "generate_grouped" not in source
    assert "verifier" not in source.lower()
    assert "ref_policy_wg" not in source


def _rl_batch():
    prompts = torch.tensor([[0, 10], [0, 11], [0, 12], [0, 13]])
    responses = torch.tensor([[20, 2], [21, 2], [22, 2], [23, 2]])
    attention = torch.tensor([[0, 1, 1, 1]] * 4)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention,
            "position_ids": torch.tensor([[0, 0, 1, 2]] * 4),
            "response_mask": torch.ones(4, 2, dtype=torch.long),
            "old_log_probs": torch.zeros(4, 2),
            "advantages": torch.ones(4, 2),
        }
    )


def test_augmented_update_keeps_optimizer_mini_batch_count_and_ppo_token_denominator():
    rl = _rl_batch()
    support = build_support_batch(
        prompt_tokens_by_key={"a": [5], "b": [6]},
        witnesses=[
            {"prompt_key": "a", "response_token_ids": [30, 2], "reference_seq_logprob": -2.0},
            {"prompt_key": "b", "response_token_ids": [31, 2], "reference_seq_logprob": -2.0},
        ],
        prompt_width=2,
        response_width=2,
        pad_token_id=0,
    )
    augmented = build_augmented_actor_batch(
        rl,
        support,
        lambda_value=0.1,
        alpha=0.5,
        global_support_batch_size=2,
    )
    captured = {}

    class Worker:
        def update_actor(self, data):
            captured["num_mini_batch"] = tu.get_non_tensor_data(data, "num_mini_batch", None)
            captured["mini_batch_size"] = tu.get_non_tensor_data(data, "mini_batch_size", None)
            captured["loss_tokens"] = int(data["loss_mask"].sum().item())
            return tu.get_tensordict(
                {},
                non_tensor_dict={"metrics": {"mfu": 0.0}},
            )

    trainer = object.__new__(RayDAPOSuccessSupportFloorTrainer)
    trainer.global_steps = 2
    trainer.actor_rollout_wg = Worker()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=ActorConfig(
                strategy="fsdp",
                rollout_n=2,
                use_dynamic_bsz=True,
                ppo_mini_batch_size=1,
                ppo_epochs=1,
                loss_agg_mode="token-mean",
            ),
            rollout=SimpleNamespace(n=2, temperature=1.0),
        )
    )
    trainer._update_actor_augmented(augmented, rl_batch_size=4)

    assert captured["num_mini_batch"] == 2
    assert captured["mini_batch_size"] is None
    assert captured["loss_tokens"] == int(rl.batch["response_mask"].sum().item())


def test_launchers_are_syntax_valid_and_keep_shadow_before_dual():
    example_dir = ROOT / "examples" / "success_support_floor"
    smoke = example_dir / "train_dapo_qwen3_4b_h4096_bssf_smoke.sh"
    active = example_dir / "train_dapo_qwen3_4b_h4096_bssf.sh"
    subprocess.run(["bash", "-n", str(smoke)], check=True)
    subprocess.run(["bash", "-n", str(active)], check=True)
    smoke_text = smoke.read_text()
    active_text = active.read_text()
    assert "algorithm.success_support_floor.mode=shadow" in smoke_text
    assert "trainer.total_training_steps=2" in smoke_text
    assert "algorithm.success_support_floor.mode=dual" in active_text
    assert "trainer.total_training_steps=200" in active_text
    assert "data.max_response_length=4096" in smoke_text
    assert "data.max_response_length=4096" in active_text
    assert "generate_grouped" not in smoke_text + active_text
    assert "sbatch" not in smoke_text + active_text
    assert "srun" not in smoke_text + active_text
