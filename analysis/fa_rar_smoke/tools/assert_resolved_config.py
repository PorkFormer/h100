#!/usr/bin/env python3
"""Fail closed unless a resolved Hydra config is the four-step FA-RAR smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    checks = {
        "algorithm.adv_estimator": (config.algorithm.adv_estimator, "grpo"),
        "algorithm.censor_aware_advantage.enable": (
            config.algorithm.censor_aware_advantage.enable,
            True,
        ),
        "algorithm.censor_aware_advantage.apply": (
            config.algorithm.censor_aware_advantage.apply,
            True,
        ),
        "algorithm.censor_aware_advantage.mode": (
            config.algorithm.censor_aware_advantage.mode,
            "reliability_redistribution",
        ),
        "actor.loss_agg_mode": (config.actor_rollout_ref.actor.loss_agg_mode, "token-mean"),
        "forced_answer_probe.enable": (
            config.actor_rollout_ref.rollout.forced_answer_probe.enable,
            True,
        ),
        "forced_answer_probe.num_samples": (
            config.actor_rollout_ref.rollout.forced_answer_probe.num_samples,
            2,
        ),
        "forced_answer_probe.training_credit.enable": (
            config.actor_rollout_ref.rollout.forced_answer_probe.training_credit.enable,
            False,
        ),
        "algorithm.probe_credit.enable": (config.algorithm.probe_credit.enable, False),
        "data.max_response_length": (config.data.max_response_length, 2048),
        "rollout.response_length": (config.actor_rollout_ref.rollout.response_length, 2048),
        "rollout.max_model_len": (config.actor_rollout_ref.rollout.max_model_len, 4096),
        "data.train_batch_size": (config.data.train_batch_size, 2),
        "data.gen_batch_size": (config.data.gen_batch_size, 2),
        "rollout.n": (config.actor_rollout_ref.rollout.n, 4),
        "trainer.total_training_steps": (config.trainer.total_training_steps, 4),
        "trainer.val_before_train": (config.trainer.val_before_train, False),
        "trainer.save_freq": (config.trainer.save_freq, -1),
        "trainer.test_freq": (config.trainer.test_freq, -1),
        "trainer.project_name": (config.trainer.project_name, "h100_verl"),
        "trainer.experiment_name": (config.trainer.experiment_name, args.run_name),
        "trainer.default_local_dir": (
            str(Path(config.trainer.default_local_dir).resolve()),
            str(args.output_dir.resolve()),
        ),
    }
    failures = [
        f"{name}: actual={actual!r} expected={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    loggers = list(config.trainer.logger)
    if "wandb" not in loggers:
        failures.append(f"trainer.logger: wandb missing from {loggers!r}")
    if failures:
        raise SystemExit("FA-RAR resolved-config gate failed:\n" + "\n".join(failures))

    print("FA-RAR resolved-config gate: PASS")
    for name, (actual, _) in checks.items():
        print(f"{name}={actual}")
    print(f"trainer.logger={loggers}")
    print("mode_dispatch=fa_rar_only")
    print("training_credit_reward_correction=inactive")


if __name__ == "__main__":
    main()
