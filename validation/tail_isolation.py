"""Controlled GPU loss diagnostic; synthetic score changes are not verifier results."""
import json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
from verl.trainer.ppo.core_algos import get_policy_loss_fn,compute_grpo_outcome_advantage
from verl.workers.config.actor import ActorConfig
record=torch.load(ROOT/'evidence/actor_h2048/rank0_gspo_off_repeat0.pt',weights_only=False)['inputs'][0]
cfg=ActorConfig(strategy='fsdp',rollout_n=4,ppo_micro_batch_size_per_gpu=1)
base=record['log_prob'].cuda();old=record['old'].cuda();adv=record['advantage'].cuda();mask=record['mask'].cuda()
frozen=torch.load(ROOT/'evidence/replay_h2048_v2/actor_input.pt',weights_only=False)
batch=frozen['batch'][:8];changed=frozen['scores']['off'][:8].clone();changed[0,int(batch.batch['response_mask'][0].sum())-1]+=1
changed_adv,_=compute_grpo_outcome_advantage(changed,batch.batch['response_mask'],batch.non_tensor_batch['uid'])
def evaluate(tail_value,score_multiplier):
    lp=torch.cat([base,torch.full((1,64),tail_value,device='cuda')],-1).requires_grad_();ol=torch.cat([old,torch.zeros(1,64,device='cuda')],-1);ad=torch.cat([adv if score_multiplier==1 else changed_adv[0:1].cuda(),torch.zeros(1,64,device='cuda')],-1);ma=torch.cat([mask,torch.zeros(1,64,device='cuda')],-1)
    loss,_=get_policy_loss_fn('gspo')(old_log_prob=ol,log_prob=lp,advantages=ad,response_mask=ma,config=cfg);loss.backward();return loss.detach(),lp.grad.detach()
a,g=evaluate(-1.,1.);b,h=evaluate(-100.,1.);c,k=evaluate(-1.,2.)
assert torch.equal(a,b) and torch.equal(g,h);assert torch.count_nonzero(g[:,-64:])==0
assert not torch.equal(g,k),'controlled advantage change must affect prefix gradient'
assert torch.isfinite(k).all()
torch.save(dict(loss=a,gradient=g,changed_tail_gradient=h,controlled_changed_advantage_gradient=k),ROOT/'evidence/tail_isolation.pt')
(ROOT/'evidence/tail_isolation.json').write_text(json.dumps(dict(status='PASS',scope='Actual saved GPU logprobs and registered GSPO; synthetic single-row task score increment and original GRPO test score sensitivity, not a real verifier transition'),indent=2))
