"""Run an already migrated, resolved config without repeating legacy migration."""
import argparse,sys,os
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def validate_resolved(config):
    from verl.experimental.natural_continuation_boundary_return.config import map_ncbr_config
    from verl.experimental.natural_continuation_boundary_return.validation import validate_boundary_return_preflight
    from verl.utils.config import omega_conf_to_dataclass,validate_config
    from verl.trainer.ppo.utils import need_reference_policy,need_critic
    assert not any(k in config for k in ['reward_model','custom_reward_function','sandbox_fusion']), 'expected post-migration config'
    assert "VLLM_PORT" not in os.environ, "inherited VLLM_PORT would collide across replicas"
    assert "VLLM_PORT" not in config.ray_kwargs.ray_init.runtime_env.env_vars, "shared VLLM_PORT is forbidden"
    before=OmegaConf.to_container(config,resolve=True)
    map_ncbr_config(config)
    validate_boundary_return_preflight(config,require_dynamic_filter=False)
    validate_config(config,use_reference_policy=need_reference_policy(config),use_critic=need_critic(config))
    for key in ['actor_rollout_ref.actor','actor_rollout_ref.rollout','algorithm']:
        omega_conf_to_dataclass(OmegaConf.select(config,key))
    assert OmegaConf.to_container(config,resolve=True)==before,'resolved config changed'
    return config

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    config=validate_resolved(OmegaConf.load(a.config))
    if a.validate_only:
        print('PASS: post-migration config and typed workers validated; Ray not initialized');return
    from verl.trainer.main_ppo import run_ppo
    run=Path(config.trainer.grpo_audit_path).parent
    OmegaConf.save(config,run/'actual_resolved_config.yaml',resolve=True)
    run_ppo(config)
if __name__=='__main__':main()
