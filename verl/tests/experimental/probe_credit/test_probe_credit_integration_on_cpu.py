from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

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
from verl.trainer.ppo.core_algos import compute_grpo_vectorized_outcome_advantage

ROOT = Path(__file__).resolve().parents[3]


def test_probe_credit_dapo_config_composes_with_canonical_defaults():
    config_dir = ROOT / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="probe_credit_dapo_trainer")

    assert config.algorithm.adv_estimator == "grpo"
    assert config.algorithm.filter_groups.enable is True
    assert config.algorithm.probe_credit.enable is False
    assert config.algorithm.probe_credit.coef == 0.0
    assert config.actor_rollout_ref.rollout.name == "vllm"


def test_two_prompt_four_rollout_cpu_pipeline_preserves_science_invariants():
    uids = np.asarray(["p0"] * 4 + ["p1"] * 4 + ["filtered"] * 4, dtype=object)
    trajectory_ids = np.asarray([f"t{i}" for i in range(12)], dtype=object)
    terminal = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    token_scores = torch.zeros(12, 10)
    token_scores[:, -1] = terminal
    candidate = DataProto.from_dict(
        tensors={"token_level_scores": token_scores, "token_level_rewards": token_scores.clone()},
        non_tensors={"uid": uids, "trajectory_id": trajectory_ids, "acc": terminal.numpy()},
    )

    retained = select_complete_prompt_groups(filter_dapo_generation_batch(candidate, "acc"), 2, 4)
    retained_ids = retained.non_tensor_batch["trajectory_id"].copy()
    rewards_before = retained.batch["token_level_rewards"].clone()
    lengths = [10, 9, 8, 7, 10, 9, 8, 7]
    response_mask = torch.zeros(8, 10)
    for row, length in enumerate(lengths):
        response_mask[row, :length] = 1
    terminal_advantages, terminal_returns = compute_grpo_vectorized_outcome_advantage(
        retained.batch["token_level_rewards"], response_mask, retained.non_tensor_batch["uid"]
    )
    retained.batch["advantages"] = terminal_advantages
    retained.batch["returns"] = terminal_returns
    values = torch.tensor(
        [
            [0, 0, 0, 0.75, 0.75],
            [0, 0.25, 0.5, 1, 1],
            [0, 0, 0.25, 0.25, 0.5],
            [0, 0.5, 0.5, 0.75, 1],
            [0.25, 0.25, 0.5, 0.5, 0.75],
            [0.25, 0.5, 0.5, 0.75, 0.75],
            [0.25, 0.25, 0.25, 0.5, 1],
            [0.25, 0.5, 0.75, 1, 1],
        ]
    )
    temporal = compute_probe_temporal_returns(compute_probe_pseudo_rewards(values), rho=0.5)
    segment_advantages, _ = compute_group_relative_probe_advantages(temporal, retained.non_tensor_batch["uid"])
    correction, correction_metrics = build_probe_token_correction(segment_advantages, response_mask)
    apply_probe_credit_redistribution(retained, correction, coef=0.2, enable=True)

    assert retained.non_tensor_batch["trajectory_id"].tolist() == retained_ids.tolist()
    assert torch.equal(retained.batch["token_level_rewards"], rewards_before)
    assert torch.equal(retained.batch["token_level_scores"], rewards_before)
    assert torch.equal(retained.batch["terminal_advantages"], terminal_advantages)
    assert torch.equal(retained.batch["returns"], retained.batch["advantages"])
    torch.testing.assert_close((correction * response_mask).sum(-1), torch.zeros(8), atol=1e-6, rtol=0)
    for row, length in enumerate(lengths):
        tail = 9 * length // 10
        assert torch.equal(correction[row, tail:], torch.zeros(10 - tail))
    assert correction_metrics["probe_credit/zero_mass_residual_max"] <= 1e-6


def test_entrypoint_and_launcher_are_dedicated_and_do_not_submit_slurm():
    entrypoint = (ROOT / "verl" / "experimental" / "probe_credit" / "main_dapo_probe_credit.py").read_text()
    launcher = (ROOT / "examples" / "probe_credit" / "train_dapo_qwen3_8b_h100x8_probe_credit_smoke.sh").read_text()

    assert "RayDAPOProbeCreditTrainer" in entrypoint
    assert "RayPPOTrainer(" not in entrypoint
    assert "PROBE_CREDIT_COEF" in launcher
    assert "sbatch" not in launcher and "srun" not in launcher
