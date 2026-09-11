"""Frozen real token replay through source adapter and extracted hook."""
import argparse,copy,dataclasses,importlib.util,json,subprocess,sys,types
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
from verl import DataProto
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.workers.reward_manager.dapo import DAPORewardManager
from verl.experimental.natural_continuation_boundary_return.hook import NCBRHook
from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations,build_continuation_requests
from verl.experimental.natural_continuation_boundary_return.scoring import score_long_generations
from verl.experimental.natural_continuation_boundary_return.reward_adapter import BoundaryRewardOutput
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--suffix',default='v2');p.add_argument('--block',type=int,default=None);p.add_argument('--h',type=int,default=128);p.add_argument('--l',type=int,default=512);a=p.parse_args();tag='' if a.block is None else f'_b{a.block:04d}';out=ROOT/'evidence'/f'replay_h{a.h}{tag}_{a.suffix}';out.mkdir(exist_ok=False)
def oracle_module(name,filename):
    source=subprocess.check_output(['git','show',f'd23da0e:verl/verl/experimental/natural_continuation_boundary_return/{filename}.py'],cwd=ROOT.parent,text=True)
    (out/f'oracle_{filename}.py.txt').write_text(source)
    mod=types.ModuleType(name);mod.__file__=str(out/f'oracle_{filename}.py.txt');sys.modules[name]=mod;exec(compile(source,mod.__file__,'exec'),mod.__dict__);return mod
oracle=oracle_module('validation_oracle_adapter','reward_adapter')
oracle_runtime=oracle_module('validation_oracle_runtime','runtime')
tok=AutoTokenizer.from_pretrained('/workspace/models/Qwen3-1.7B-Base',local_files_only=True)
rows=[];requests=[];saved={};repeat_diffs=[]
for replica in range(8):
    folder=ROOT/'evidence'/f'infer_h{a.h}{tag}_replica{replica}'
    rows.extend(json.loads((folder/'short.json').read_text()))
    rep0=json.loads((folder/'repeat_0_continuations.json').read_text());rep1=json.loads((folder/'repeat_1_continuations.json').read_text())
    repeat_diffs.extend([x['request_id'] for x,y in zip(rep0,rep1,strict=True) if x!=y]);saved.update({r['request_id']:r for r in rep0})
rows.sort(key=lambda r:(r['index'],r['trajectory_id']));n=len(rows);width=max(len(r['prompt_token_ids']) for r in rows)
pt=torch.full((n,width),tok.pad_token_id,dtype=torch.long);rt=torch.full((n,a.h),tok.pad_token_id,dtype=torch.long);pm=torch.zeros_like(pt);rm=torch.zeros_like(rt)
for i,r in enumerate(rows):
    ids=r['prompt_token_ids'];t=r['tokens'];pt[i,-len(ids):]=torch.tensor(ids);pm[i,-len(ids):]=1;rt[i,:len(t)]=torch.tensor(t);rm[i,:len(t)]=1
attn=torch.cat([pm,rm],-1)
batch=DataProto.from_dict(tensors=dict(prompts=pt,responses=rt,response_mask=rm,input_ids=torch.cat([pt,rt],-1),attention_mask=attn,position_ids=(attn.cumsum(-1)-1).clamp(min=0)),non_tensors=dict(uid=np.array([r['uid'] for r in rows],dtype=object),trajectory_id=np.array([r['trajectory_id'] for r in rows],dtype=object),rollout_policy_version=np.array([0]*n,dtype=object),finish_reason=np.array([r['finish_reason'] for r in rows],dtype=object),data_source=np.array(['math_dapo']*n,dtype=object),reward_model=np.array([dict(ground_truth=r['ground_truth']) for r in rows],dtype=object)))
manager=DAPORewardManager(tok,0,max_resp_len=a.h,overlong_buffer_cfg=OmegaConf.create(dict(enable=True,len=32 if a.h==128 else 410,penalty_factor=1,log=True)))
def score(b):
    result=manager(copy.deepcopy(b),return_dict=True)
    return BoundaryRewardOutput(result['reward_tensor'],{k:np.array(v) for k,v in result['reward_extra_info'].items()})
short=score(batch);raw=short.reward_tensor.clone();extras=short.extra_info
batch.batch['token_level_scores']=raw.clone();batch.batch['token_level_rewards']=raw.clone();batch.non_tensor_batch.update(extras)
kwargs=dict(policy_version=0,short_response_length=a.h,long_response_length=a.l,max_model_len=9216,eos_token_id=tok.eos_token_id,base_seed=42)
r1,m1=build_continuation_requests(batch,**kwargs);r2,m2=oracle_runtime.build_continuation_requests(batch,**kwargs)
assert [dataclasses.asdict(r) for r in r1]==[dataclasses.asdict(r) for r in r2];assert np.array_equal(m1,m2)
class FrozenClient:
    def __init__(self):self.events=[]
    async def start_grouped(self,request_id,**kw):
        self.events.append(dict(event='start',request_id=request_id,**kw));s=saved[request_id];client=self
        class Handle:
            backend_request_id=request_id;server_id='frozen-not-remote'
            async def result(self):return [types.SimpleNamespace(token_ids=s['tokens'],stop_reason=s['stop_reason'] or s['finish_reason'],extra_fields=dict(branch_id=0,global_steps=0,finish_reason=s['finish_reason']))]
            async def release(self):client.events.append(dict(event='release',request_id=request_id))
            async def abort(self):client.events.append(dict(event='abort',request_id=request_id))
            async def drain(self):client.events.append(dict(event='drain',request_id=request_id))
        return Handle()
padding=[];long_verifier=[]
def long_score(work,generations,cfg):
    def cb(b):
        padding.append(len(b));assert len(b)%2==0;result=score(b);long_verifier.append(dict(rows=len(b),extras={k:v.tolist() for k,v in result.extra_info.items()}));return result
    return score_long_generations(work,generations,cfg,score_batch=cb,pad_token_id=tok.pad_token_id,reward_worker_count=2)
results={};scores={}
for mode in ['off','shadow','replace']:
    cfg=BoundaryReturnConfig(mode=mode,long_response_length=a.l,max_concurrent_requests=4,request_batch_size=8,long_reward_chunk_size=3,seed=42)
    common=dict(config=cfg,policy_version=0,sampling_params=dict(temperature=1,top_p=1,top_k=-1,ignore_eos=False),eos_token_id=tok.eos_token_id,short_response_length=a.h,max_model_len=9216)
    c=FrozenClient();outcome=NCBRHook().apply(batch,raw,raw_verifier_extras=extras,continuation_client=c,score_long=long_score,**common)
    original=copy.deepcopy(batch);oc=FrozenClient()
    if mode!='off':
        cap=oracle_runtime.run_boundary_continuations(rollout_batch=original,client=oc,**common)
        lr=long_score(original,cap.generations,cfg) if cap.generations else BoundaryRewardOutput(torch.empty((0,0)),dict(acc=np.array([]),score=np.array([])))
        original_result=oracle.apply_boundary_return(original,capture=cap,long_reward_output=lr,config=cfg)
    assert torch.equal(outcome.effective_scores,original.batch['token_level_scores']);assert c.events==oc.events
    assert torch.equal(batch.batch['token_level_scores'],raw)
    delta=outcome.effective_scores-raw;allowed=torch.zeros_like(delta,dtype=torch.bool);allowed[torch.arange(n),rm.sum(-1)-1]=True;assert torch.equal(delta[~allowed],torch.zeros_like(delta[~allowed]))
    from verl.experimental.probe_credit.dynamic_sampling import filter_dapo_generation_batch
    hook_batch=copy.deepcopy(batch)
    if mode!='off':
        result=outcome.aux['result']
        for key in ['hit_response_cap','short_acc','long_acc','boundary_acc','task_score_delta','uids']:
            assert np.array_equal(getattr(result,key),getattr(original_result,key),equal_nan=True) if key!='uids' else np.array_equal(result.uids,original_result.uids)
        for key,value in outcome.aux['row_labels'].items():assert torch.equal(value,original.batch[key]),key
        if mode=='replace':
            hook_batch.non_tensor_batch['acc']=result.boundary_acc
            original.non_tensor_batch['acc']=original_result.boundary_acc
        (out/f'{mode}_transitions.json').write_text(json.dumps(dict(short=result.short_acc.tolist(),long=result.long_acc.tolist(),hit_cap=result.hit_response_cap.tolist())))
    kept_hook=filter_dapo_generation_batch(hook_batch,'acc');kept_source=filter_dapo_generation_batch(original,'acc')
    assert kept_hook.non_tensor_batch['trajectory_id'].tolist()==kept_source.non_tensor_batch['trajectory_id'].tolist()
    (out/f'{mode}_retained.json').write_text(json.dumps(kept_hook.non_tensor_batch['trajectory_id'].tolist()))
    scores[mode]=outcome.effective_scores;results[mode]=dict(exact=True,calls=len(c.events),changed_rows=int((delta!=0).any(-1).sum()))
    (out/f'{mode}_calls.json').write_text(json.dumps(c.events))
torch.save(dict(batch=batch,raw=raw,extras=extras,scores=scores),out/'actor_input.pt')
(out/'long_verifier.json').write_text(json.dumps(long_verifier))
(out/'short_verifier.json').write_text(json.dumps({k:v.tolist() for k,v in extras.items()}))
(out/'summary.json').write_text(json.dumps(dict(status='PASS',modes=results,rows=n,cap_hits=int(m1.sum()),independent_repeat_differences=repeat_diffs,padded_long_chunk_sizes=padding,scope='Frozen runtime and adapter equivalence only; verifier executed locally, 2-worker divisibility checked, no remote worker/release attestation'),indent=2))
