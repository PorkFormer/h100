"""Diagnostic entrypoint for exactly one frozen DAPO actor update."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from pprint import pprint

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import hydra
import ray
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl.experimental.on_policy_budgeted_capability_floor.frozen_batch_trainer import (
    RayDAPOFrozenBatchBaselineTrainer,
    RayDAPOFrozenBatchOBCFTrainer,
)
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, create_rl_sampler, run_ppo
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


class FrozenPlaceholderDataset(Dataset):
    """Non-iterated shape placeholder for the diagnostic trainer constructor."""

    def __init__(self, length: int):
        if int(length) <= 0:
            raise ValueError("frozen placeholder dataset length must be positive")
        self._length = int(length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        raise RuntimeError("frozen-batch harness must never iterate its placeholder dataset")


class FrozenBatchTaskRunner(TaskRunner):
    """Initialize the unchanged worker topology and execute the diagnostic adapter."""

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        print(f"FrozenBatchTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.resolve(config)
        raw = config.trainer.get("frozen_batch")
        if raw is None:
            raise ValueError("trainer.frozen_batch configuration is required")
        mode = str(raw.get("mode", ""))
        if mode == "baseline":
            trainer_class = RayDAPOFrozenBatchBaselineTrainer
        elif mode in ("off", "shadow"):
            trainer_class = RayDAPOFrozenBatchOBCFTrainer
        else:
            raise ValueError(f"invalid frozen-batch mode: {mode!r}")

        _actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_reward_model_resource_pool(config)
        validate_config(config=config, use_reference_policy=False, use_critic=False)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)
        # Frozen replay never reads a dataloader.  Materializing the scientific
        # parquet datasets would only repeat their expensive filter pass and
        # cannot affect the already attested DataProto supplied to the harness.
        generation_batch_size = config.data.get(
            "gen_batch_size", config.data.train_batch_size
        )
        train_dataset = FrozenPlaceholderDataset(generation_batch_size)
        val_dataset = FrozenPlaceholderDataset(1)
        trainer = trainer_class(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
        )
        trainer._obcf_base_model_local_path = local_path
        trainer.init_workers()
        trainer.fit()


@hydra.main(
    config_path="../../trainer/config",
    config_name="on_policy_budgeted_capability_floor_dapo_trainer",
    version_base=None,
)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    runner_class = ray.remote(num_cpus=1)(FrozenBatchTaskRunner)
    run_ppo(config, task_runner_class=runner_class)


if __name__ == "__main__":
    main()
