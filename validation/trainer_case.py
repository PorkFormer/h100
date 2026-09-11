"""Isolated actual trainer entry; configuration and worker receipts remain in the case directory."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
p=argparse.ArgumentParser();p.add_argument('--entry',choices=['dapo','standard'],required=True);p.add_argument('--mode',choices=['off','shadow','replace'],required=True);p.add_argument('--loss',choices=['vanilla','gspo'],default='vanilla');p.add_argument('--ref',action='store_true');p.add_argument('--live-oracle-smoke',action='store_true');p.add_argument('--live-oracle-full',action='store_true');p.add_argument('--fault',default=None,choices=['missing','duplicate','version','verifier_error','verifier_timeout','release']);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
case=ROOT/'evidence'/f'train_{a.entry}_{a.loss}_{a.mode}{"_ref" if a.ref else ""}{("_fault_"+a.fault) if a.fault else ""}{"_live_oracle_smoke" if a.live_oracle_smoke else ""}{"_live_oracle_full" if a.live_oracle_full else ""}_v6'
case.mkdir(exist_ok=True)
with initialize_config_dir(config_dir=str(ROOT.parent/'verl/verl/trainer/config'),version_base=None):cfg=compose(config_name='natural_continuation_boundary_return_dapo_trainer' if a.entry=='dapo' else 'ppo_trainer')
settings={
'algorithm.adv_estimator':'grpo','algorithm.use_kl_in_reward':False,'algorithm.rollout_correction':None,'critic.enable':False,'distillation.enabled':False,
'data.train_files':str(ROOT/'evidence/prompts.parquet'),'data.val_files':str(ROOT/'evidence/prompts.parquet'),'data.max_prompt_length':1024,'data.max_response_length':2048,'data.train_batch_size':2,'data.shuffle':False,'data.dataloader_num_workers':0,
'actor_rollout_ref.model.path':'/workspace/models/Qwen3-1.7B-Base','actor_rollout_ref.model.enable_gradient_checkpointing':True,
'actor_rollout_ref.actor.strategy':'fsdp','actor_rollout_ref.actor.ppo_mini_batch_size':2,'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,'actor_rollout_ref.actor.ppo_epochs':1,'actor_rollout_ref.actor.optim.lr':1e-6,'actor_rollout_ref.actor.use_kl_loss':a.ref,'actor_rollout_ref.actor.kl_loss_coef':0.,'actor_rollout_ref.actor.policy_loss.loss_mode':a.loss,'actor_rollout_ref.actor.fsdp_config.use_torch_compile':False,
'actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.n':4,'actor_rollout_ref.rollout.tensor_model_parallel_size':1,'actor_rollout_ref.rollout.gpu_memory_utilization':.35,'actor_rollout_ref.rollout.enforce_eager':True,'actor_rollout_ref.rollout.max_model_len':9216,'actor_rollout_ref.rollout.temperature':1.,'actor_rollout_ref.rollout.top_p':1.,'actor_rollout_ref.rollout.top_k':-1,'actor_rollout_ref.rollout.ignore_eos':False,'actor_rollout_ref.rollout.engine_kwargs.vllm.seed':42,'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1,'actor_rollout_ref.rollout.agent.num_workers':8,
'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu':1,
'actor_rollout_ref.rollout.boundary_return.mode':a.mode,'actor_rollout_ref.rollout.boundary_return.long_response_length':8192,'actor_rollout_ref.rollout.boundary_return.max_concurrent_requests':4,'actor_rollout_ref.rollout.boundary_return.request_batch_size':8,'actor_rollout_ref.rollout.boundary_return.request_timeout_seconds':600.,'actor_rollout_ref.rollout.boundary_return.long_reward_chunk_size':3,'actor_rollout_ref.rollout.boundary_return.seed':42,
'reward.num_workers':2,'reward.reward_manager.name':'dapo','reward.reward_kwargs.max_resp_len':2048,'reward.reward_kwargs.overlong_buffer_cfg':dict(enable=True,len=410,penalty_factor=1.,log=True),
'ncbr.enable':None if a.entry=='dapo' and a.mode=='off' else a.mode!='off',
'trainer.n_gpus_per_node':8,'trainer.nnodes':1,'trainer.total_training_steps':1 if a.ref else 2,'trainer.total_epochs':1,'trainer.val_before_train':False,'trainer.test_freq':-1,'trainer.save_freq':-1,'trainer.logger':['console'],'trainer.project_name':'ncbr_validation','trainer.experiment_name':case.name,'trainer.default_local_dir':str(case/'artifacts'),'trainer.resume_mode':'disable',
'ray_kwargs.ray_init.address':'local','ray_kwargs.ray_init._temp_dir':str(ROOT.parent/'r'/__import__('hashlib').sha256(case.name.encode()).hexdigest()[:6]),'ray_kwargs.ray_init.namespace':case.name,'ray_kwargs.ray_init.num_cpus':48,'ray_kwargs.ray_init.include_dashboard':False,
}
if a.entry=='dapo':settings.update({'data.gen_batch_size':4,'algorithm.filter_groups.max_num_gen_batches':4})
if a.fault or a.live_oracle_smoke:settings.update({'data.max_response_length':128,'actor_rollout_ref.rollout.boundary_return.long_response_length':512,'reward.reward_kwargs.max_resp_len':128,'reward.reward_kwargs.overlong_buffer_cfg':dict(enable=True,len=32,penalty_factor=1.,log=True),'trainer.total_training_steps':1})
if a.live_oracle_smoke:settings['data.train_batch_size']=8
if a.live_oracle_full:settings.update({'data.train_batch_size':32,'trainer.total_training_steps':1})
for k,v in settings.items():OmegaConf.update(cfg,k,v,force_add=True)
(case/'config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
if a.dry_run:
    from verl.utils.config import omega_conf_to_dataclass
    for item in [cfg.actor_rollout_ref.actor,cfg.actor_rollout_ref.rollout]:omega_conf_to_dataclass(item)
    sys.exit(0)
import ray
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
try:
    cfg=migrate_legacy_reward_impl(cfg)
    if a.entry=='dapo':
        from verl.experimental.natural_continuation_boundary_return.main_dapo_boundary_return import BoundaryReturnTaskRunner
        run_ppo(cfg,task_runner_class=ray.remote(num_cpus=1)(BoundaryReturnTaskRunner))
    else:run_ppo(cfg)
finally:
    ray.shutdown()
