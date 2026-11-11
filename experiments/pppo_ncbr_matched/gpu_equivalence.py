"""Real 16-rank native FSDP BF16 gate; never substitutes FP32 for acceptance."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from prepare import CONFIG, REPO, RUN, sha, write_json, code_fingerprint


def compare_vectors(reference, actual):
    import torch
    reference, actual = reference.double(), actual.double()
    norm = torch.linalg.vector_norm(reference)
    error = torch.linalg.vector_norm(actual-reference)
    relative = float(error/norm) if norm else (0.0 if not error else float('inf'))
    cosine = float(torch.nn.functional.cosine_similarity(reference,actual,dim=0)) if norm else None
    return dict(relative_error=relative, cosine=cosine, reference_norm=float(norm),
                actual_norm=float(torch.linalg.vector_norm(actual)), finite=bool(torch.isfinite(actual).all()))


class FrozenRank:
    def run(self, rank, address, config_path, output):
        import numpy as np
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.experimental.pppo.objective import group_statistics, pppo_loss
        from verl.experimental.pppo.prefix import actor_batch
        from verl.utils import tensordict_utils as tu
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig
        from verl.trainer.config import CheckpointConfig
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        torch.cuda.set_device(0)
        dist.init_process_group('nccl',init_method=address,rank=rank,world_size=16,timeout=timedelta(minutes=15))
        c=OmegaConf.load(config_path)
        model=omega_conf_to_dataclass(c.actor_rollout_ref.model,dataclass_type=HFModelConfig)
        # Match the actual worker's Hydra _target_ factory, not structured schema
        # construction (legacy engine fields permit None defaults at runtime).
        ec=omega_conf_to_dataclass(c.actor_rollout_ref.actor.fsdp_config)
        oc=omega_conf_to_dataclass(c.actor_rollout_ref.actor.optim)
        oc.total_training_steps=300
        cc=omega_conf_to_dataclass(c.actor_rollout_ref.actor.checkpoint)
        engine=FSDPEngineWithLMHead(model,ec,oc,cc)
        engine.initialize()
        engine.pppo_strict_update=True
        tokenizer=model.tokenizer
        prompt=tokenizer.encode('Solve the following math problem step by step. What is 17 times 23?\nAnswer:',add_special_tokens=False)
        text=tokenizer.encode('We multiply 17 by 23. First compute 17 times 20, which is 340. Then add 51. The answer is 391.\n',add_special_tokens=False)
        lengths=torch.tensor([32,63,127,511,1024,2048,4096,8192])
        width=8192
        h=lengths*15//100
        prompts=torch.tensor([[tokenizer.pad_token_id]*3+prompt]*8)
        responses=torch.full((8,width),tokenizer.pad_token_id,dtype=torch.long)
        for i,length in enumerate(lengths):
            responses[i,:length]=torch.tensor((text*((int(length)//len(text))+1))[:length])
        response_mask=torch.arange(width)[None,:]<lengths[:,None]
        pm=torch.tensor([[0]*3+[1]*len(prompt)]*8)
        attention=torch.cat((pm,response_mask.long()),-1)
        reward=torch.tensor([0.,1.,2.,3.,4.,5.,7.,9.])
        uids=[str(rank)]*8
        advantage,total,weight=group_statistics(reward,lengths,uids)
        original=DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,response_mask=response_mask,
             input_ids=torch.cat((prompts,responses),-1),attention_mask=attention,position_ids=(attention.cumsum(-1)-1).clamp(min=0),
             pppo_prefix_length=h,pppo_original_length=lengths,pppo_reward=reward,pppo_advantage=advantage,
             pppo_group_total_length=total,pppo_row_weight=weight),non_tensors=dict(uid=np.array(uids,dtype=object)))
        parameters=list(engine.module.parameters())
        initial=[p.detach().cpu().clone() for p in parameters]
        results={}
        suffix_gradient_max=[]
        directory=Path(output)
        directory.mkdir(parents=True,exist_ok=True)

        def loss_fn(model_output,data,dp_group=None):
            def check_suffix(gradient):
                padded=no_padding_2_padding(gradient,data)
                outside=~data['pppo_loss_mask'].bool()
                suffix_gradient_max.append(float(padded[outside].abs().max()) if outside.any() else 0.)
                return gradient
            model_output['log_probs'].register_hook(check_suffix)
            return pppo_loss(c.actor_rollout_ref.actor,model_output,data)

        # Compare all forward/backward paths before any optimizer mutation. A
        # same-view repeat separates stochastic kernels from length effects.
        for view in ['full_mask_reference','cropped','full_mask_repeat']:
            with torch.no_grad():
                for parameter, value in zip(parameters,initial,strict=True): parameter.copy_(value.to(parameter.device))
            engine.optimizer.state.clear()
            engine.optimizer_zero_grad()
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            batch=actor_batch(copy.deepcopy(original),'full_mask_reference' if view=='full_mask_repeat' else view,tokenizer.pad_token_id)
            td=left_right_2_no_padding(batch.to_tensordict()).to('cuda')
            tu.assign_non_tensor(td,temperature=1.,use_remove_padding=True,use_dynamic_bsz=False,
                                micro_batch_size_per_gpu=2,global_batch_size=128,use_fused_kernels=False)
            with engine.eval_mode():
                inferred=engine.infer_batch(td)
            old=no_padding_2_padding(inferred['model_output']['log_probs'],td).detach()
            td['old_log_probs']=old
            tu.assign_non_tensor(td,micro_batch_size_per_gpu=1)
            with engine.train_mode():
                engine.optimizer_zero_grad()
                output_batch=engine.forward_backward_batch(td,loss_fn,forward_only=False)
                gradient=torch.cat([p.grad.detach().flatten().float().cpu() for p in parameters if p.grad is not None])
                current=no_padding_2_padding(output_batch['model_output']['log_probs'],td).detach()
                mask=batch.batch['pppo_loss_mask'].to(current.device)
                loss=sum(output_batch['loss'])
                results[view]=dict(old=old[mask].cpu(),current=current[mask].cpu(),gradient=gradient,
                                   gradient_shards=[None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters],
                                   loss=loss,denominator=total.clone())
            torch.cuda.synchronize()
        for view, result in results.items():
            with torch.no_grad():
                for parameter,value in zip(parameters,initial,strict=True):parameter.copy_(value.to(parameter.device))
            engine.optimizer.state.clear()
            with engine.train_mode():
                engine.optimizer_zero_grad()
                for parameter,gradient in zip(parameters,result.pop('gradient_shards'),strict=True):
                    parameter.grad=None if gradient is None else gradient.to(parameter.device)
                result['grad_norm']=engine.optimizer_step()
                result['optimizer_steps']=engine.pppo_optimizer_updates
                result['update']=torch.cat([(p.detach().cpu()-v).flatten().float() for p,v in zip(parameters,initial,strict=True)])
        a,b=results['full_mask_reference'],results['cropped']
        gradient=compare_vectors(a['gradient'],b['gradient'])
        update=compare_vectors(a['update'],b['update'])
        repeat=results['full_mask_repeat']
        report=dict(rank=rank,host=socket.gethostname(),world_size=dist.get_world_size(),dp_size=engine.get_data_parallel_size(),
                    dtype=str(engine._autocast_dtype),master_dtype=str(parameters[0].dtype),
                    old_logprob_max_abs=float((a['old']-b['old']).abs().max()),
                    current_logprob_max_abs=float((a['current']-b['current']).abs().max()),
                    loss_relative_error=abs(a['loss']-b['loss'])/max(abs(a['loss']),1e-12),
                    gradient=gradient,update=update,
                    grad_norm_relative_error=abs(a['grad_norm']-b['grad_norm'])/max(abs(a['grad_norm']),1e-12),
                    reference_loss=a['loss'],cropped_loss=b['loss'],
                    original_lengths=lengths.tolist(),prefix_lengths=h.tolist(),
                    within_view_old_current_max_abs={name:float((value['old']-value['current']).abs().max())
                                                     for name,value in results.items()},
                    denominator_identical=bool(torch.equal(a['denominator'],b['denominator'])),
                    optimizer_steps=[a['optimizer_steps'],b['optimizer_steps']],
                    same_view_repeat=dict(logprob_max_abs=float((a['current']-repeat['current']).abs().max()),
                                          gradient=compare_vectors(a['gradient'],repeat['gradient']),
                                          update=compare_vectors(a['update'],repeat['update'])),
                    suffix_logprob_gradient_max=max(suffix_gradient_max),
                    max_memory_allocated_bytes=torch.cuda.max_memory_allocated())
        report['passed']=all([report['old_logprob_max_abs']<=.05,report['current_logprob_max_abs']<=.05,
                             report['loss_relative_error']<=.02,report['grad_norm_relative_error']<=.02,
                             gradient['finite'],gradient['relative_error']<=.02,
                             gradient['cosine'] is not None and gradient['cosine']>=.999,
                             update['finite'],update['relative_error']<=.02,
                             gradient['reference_norm']>0,update['reference_norm']>0,
                             report['suffix_logprob_gradient_max']==0.,
                             report['denominator_identical'],report['optimizer_steps']==[1,1]])
        write_json(directory/f'rank_{rank:02d}.json',report)
        dist.destroy_process_group()
        return report


def main():
    from launch import require_gates
    require_gates(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--output',type=Path,default=RUN/'acceptance/gate2')
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Use a new gate evidence directory; never overwrite failures')
    args.output.mkdir(parents=True)
    from omegaconf import OmegaConf
    c=OmegaConf.load(args.config)
    environment=dict(c.ray_kwargs.ray_init.runtime_env.env_vars)
    environment.update(PYTHONPATH=str(REPO/'verl')+':'+str(Path(__file__).parent),
                       WANDB_MODE='disabled',NCCL_DEBUG='WARN')
    ray.init(address='10.8.191.127:6395',runtime_env={'env_vars':environment},log_to_driver=True)
    actors=[]
    started=time.time()
    report=dict(gate=2,status='FAIL',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                code_fingerprint=code_fingerprint(),
                config_sha256=sha(args.config),thresholds=dict(logprob_max_abs=.05,relative_error=.02,gradient_cosine=.999))
    try:
        nodes=sorted([n for n in ray.nodes() if n['Alive']],key=lambda n:n['NodeManagerAddress'])
        if [n['NodeManagerAddress'] for n in nodes]!=['10.8.191.127','10.8.191.131'] or ray.available_resources().get('GPU')!=16:
            raise RuntimeError('Expected both allocated nodes and all sixteen available GPUs')
        with socket.socket() as s:
            s.bind(('10.8.191.127',0)); port=s.getsockname()[1]
        worker=ray.remote(num_gpus=1,num_cpus=1,max_restarts=0)(FrozenRank)
        actors=[worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[i//8]['NodeID'],soft=False)).remote()
                for i in range(16)]
        rows=ray.get([a.run.remote(i,f'tcp://10.8.191.127:{port}',str(args.config),str(args.output))
                      for i,a in enumerate(actors)],timeout=3600)
        report.update(status='PASS' if all(r['passed'] for r in rows) else 'FAIL',ranks=rows)
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        report['wall_seconds']=time.time()-started
        report['allocated_gpu_hours']=16*report['wall_seconds']/3600
        write_json(args.output/'result.json',report)
        for actor in actors: ray.kill(actor,no_restart=True)
        ray.shutdown()
    if report['status']!='PASS':raise SystemExit('BF16 equivalence failed; formal launch blocked')


if __name__=='__main__':main()
