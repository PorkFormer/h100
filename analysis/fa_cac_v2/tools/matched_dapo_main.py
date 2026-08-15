#!/usr/bin/env python3
"""Formal canonical DAPO launcher for the guarded FA-CAC v2 adapter."""

from __future__ import annotations

import os
import sys

from analysis.fa_cac_v2.tools.dapo_adapter import MatchedFACACDAPOTaskRunner, patch_resource_pool_node_affinity


def main() -> None:
    import hydra
    import ray
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from recipe.dapo.main_dapo import migrate_legacy_reward_impl, run_ppo
    from verl.utils.device import auto_set_device

    args = list(sys.argv[1:])
    show_cfg = False
    resolve = False
    overrides: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--cfg":
            if index + 1 >= len(args) or args[index + 1] != "job":
                raise SystemExit("only --cfg job is supported")
            show_cfg = True
            index += 2
            continue
        if arg == "--resolve":
            resolve = True
            index += 1
            continue
        if arg.startswith("--"):
            raise SystemExit(f"unsupported Hydra flag: {arg}")
        overrides.append(arg)
        index += 1

    os.chdir("/workspace/rl/h100-fa-cac-v2/verl")
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir="/workspace/rl/verl/recipe/dapo/config",
        job_name="fa_cac_v2_matched_dapo",
    ):
        config = hydra.compose(config_name="dapo_trainer", overrides=overrides)
    if show_cfg:
        print(OmegaConf.to_yaml(config, resolve=resolve))
        return
    patch_resource_pool_node_affinity()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    node_resource = f"node:{os.environ['FA_CAC_TARGET_NODE_IP']}"
    task_runner_class = ray.remote(num_cpus=1, resources={node_resource: 1e-4})(MatchedFACACDAPOTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
