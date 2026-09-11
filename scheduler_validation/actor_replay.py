"""Eight-rank FSDP forward/backward on the frozen H-prefix actor batch."""
import argparse,datetime,json,os,sys
from pathlib import Path
import torch,torch.distributed as dist
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,get_policy_loss_fn
from verl.workers.config.actor import ActorConfig
p=argparse.ArgumentParser();p.add_argument('--h',type=int,default=128);p.add_argument('--source');p.add_argument('--tag',default='',choices=['','_natural']);a=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=600))
source=Path(a.source) if a.source else ROOT/'evidence'/f'replay_h{a.h}_v2'/'actor_input.pt';out=Path(os.environ['NCBR_ACTOR_OUT']);out.mkdir(exist_ok=True)
frozen=torch.load(source,weights_only=False);batch=frozen['batch'];
if a.h==2048:
    changed=(frozen['scores']['replace']!=frozen['scores']['off']).any(-1)
    groups=list(dict.fromkeys(batch.non_tensor_batch['uid'].tolist()))
    changed_groups=list(dict.fromkeys(batch.non_tensor_batch['uid'][changed.numpy()].tolist()))
    chosen=(changed_groups+[g for g in groups if g not in changed_groups])[:2]
    rows=[i for i,u in enumerate(batch.non_tensor_batch['uid']) if u in chosen]
    if rank==0:(out/'selection.json').write_text(json.dumps(dict(rule='first two naturally corrected groups, fill with earliest other groups',uids=chosen,rows=rows,corrected_group_count=len(changed_groups))))
    batch=batch[rows];frozen['scores']={k:v[rows] for k,v in frozen['scores'].items()}
mask=batch.batch['response_mask'];uids=batch.non_tensor_batch['uid']
model=AutoModelForCausalLM.from_pretrained('/workspace/models/Qwen3-1.7B-Base',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True);model.config.use_cache=False;model.gradient_checkpointing_enable();model=FSDP(model,device_id=rank,mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16));model.eval()
config=ActorConfig(strategy='fsdp',rollout_n=4,ppo_micro_batch_size_per_gpu=1);evidence={};width=batch.batch['prompts'].shape[-1];assigned=list(range(rank,len(batch),8))
for name in ['vanilla','gspo']:
    for mode in ['off','shadow','replace']:
        adv,_=compute_grpo_outcome_advantage(frozen['scores'][mode],mask,uids)
        baseline=None;recorded=[]
        for repeat in range(2):
            model.zero_grad(set_to_none=True);inputs=[]
            for i in assigned:
                ids=batch.batch['input_ids'][i:i+1].cuda();attn=batch.batch['attention_mask'][i:i+1].cuda();positions=batch.batch['position_ids'][i:i+1].cuda();responses=batch.batch['responses'][i:i+1].cuda()
                logits=model(input_ids=ids,attention_mask=attn,position_ids=positions).logits[:,width-1:-1].float();lp=torch.log_softmax(logits,dim=-1).gather(-1,responses.unsqueeze(-1)).squeeze(-1)
                old=batch.batch['old_log_probs'][i:i+1].cuda() if 'old_log_probs' in batch.batch.keys() else lp.detach().clone();advantages=adv[i:i+1].cuda();response_mask=mask[i:i+1].cuda()
                loss,_=get_policy_loss_fn(name)(old_log_prob=old,log_prob=lp,advantages=advantages,response_mask=response_mask,config=config)
                assert torch.isfinite(loss);(loss/len(assigned)).backward();inputs.append(dict(row=i,old=old.cpu(),log_prob=lp.detach().cpu(),advantage=advantages.cpu(),mask=response_mask.cpu(),loss=loss.detach().cpu()))
            grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu();assert torch.isfinite(grad).all()
            torch.save(dict(inputs=inputs,gradient_shard=grad),out/f'rank{rank}_{name}_{mode}_repeat{repeat}.pt')
            if baseline is None:baseline=grad;recorded=inputs
            else:
                if not torch.equal(baseline,grad):
                    (out/f'rank{rank}_{name}_{mode}_difference.json').write_text(json.dumps(dict(exact=False,allclose_fixed=bool(torch.allclose(baseline,grad,rtol=1e-7,atol=1e-8)),max_abs=float((baseline-grad).abs().max()),different_elements=int(torch.count_nonzero(baseline!=grad)))))
                assert torch.equal(baseline,grad),f'nonexact gradient rank={rank} {name} {mode}'
                for x,y in zip(recorded,inputs,strict=True):
                    for key in ['old','log_prob','advantage','mask','loss']:assert torch.equal(x[key],y[key]),key

        evidence[f'{name}_{mode}']=dict(exact=True,grad_nonzero=int(torch.count_nonzero(baseline)))
(out/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',uuid=str(torch.cuda.get_device_properties(rank).uuid),comparisons=evidence,peak_allocated=torch.cuda.max_memory_allocated(),scope='Real FSDP repeated frozen loss/gradient determinism, no optimizer or service'),indent=2));dist.destroy_process_group()
