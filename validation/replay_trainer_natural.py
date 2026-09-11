"""Freeze the first actual version0 natural correction; rerun unchanged verifier and oracle."""
import copy,dataclasses,json,subprocess,sys,types
from pathlib import Path
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;E=R/'evidence';sys.path.insert(0,str(R.parent/'verl'))
from transformers import AutoTokenizer
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.workers.reward_manager.dapo import DAPORewardManager
from verl.experimental.natural_continuation_boundary_return.hook import NCBRHook
from verl.experimental.natural_continuation_boundary_return.reward_adapter import BoundaryRewardOutput
from verl.experimental.natural_continuation_boundary_return.scoring import score_long_generations
from verl.experimental.probe_credit.dynamic_sampling import filter_dapo_generation_batch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
record=json.loads((E/'natural_correction_first.json').read_text());assert record['policy_version']==0
p=Path(record['hook_output']);actual=torch.load(p,weights_only=False);before=torch.load(p.with_name(p.name.replace('_hook_output','_hook_input')),weights_only=False)
out=E/'replay_trainer_natural';out.mkdir(exist_ok=False)
batch=copy.deepcopy(before['batch']);raw=before['raw_scores'];extras=before['extras'];batch.batch['token_level_scores']=raw.clone();batch.batch['token_level_rewards']=raw.clone();batch.non_tensor_batch.update(extras)
tok=AutoTokenizer.from_pretrained('/workspace/models/Qwen3-1.7B-Base',local_files_only=True)
manager=DAPORewardManager(tok,0,max_resp_len=2048,overlong_buffer_cfg=OmegaConf.create(dict(enable=True,len=410,penalty_factor=1.,log=True)))
def score(b):
 x=manager(copy.deepcopy(b),return_dict=True)
 return BoundaryRewardOutput(x['reward_tensor'],{k:np.asarray(v) for k,v in x['reward_extra_info'].items()})
short=score(batch);assert torch.equal(short.reward_tensor,raw)
for k in ['acc','score']:assert np.array_equal(short.extra_info[k],np.asarray(extras[k])),k
saved={g.request_id:g for g in actual.aux['capture'].generations}
class Client:
 def __init__(self):self.events=[]
 async def start_grouped(self,request_id,**kw):
  self.events.append(dict(event='start',request_id=request_id,**kw));g=saved[request_id];parent=self
  class Handle:
   backend_request_id=request_id;server_id='frozen-real-trainer-output'
   async def result(self):return [types.SimpleNamespace(token_ids=g.tail_token_ids,stop_reason=g.stop_reason,extra_fields=dict(branch_id=0,global_steps=0,finish_reason=g.finish_reason))]
   async def release(self):parent.events.append(dict(event='release',request_id=request_id))
   async def abort(self):parent.events.append(dict(event='abort',request_id=request_id))
   async def drain(self):parent.events.append(dict(event='drain',request_id=request_id))
  return Handle()
long_evidence=[]
def long_score(work,generations,cfg):
 x=score_long_generations(work,generations,cfg,score_batch=score,pad_token_id=tok.pad_token_id,reward_worker_count=2)
 for i,g in enumerate(generations):
  assert float(x.extra_info['acc'][i])==float(actual.aux['result'].long_acc[g.parent_index])
  assert float(x.extra_info['score'][i])==float(actual.aux['result'].long_task_score[g.parent_index])
 long_evidence.append({k:np.asarray(v).tolist() for k,v in x.extra_info.items()})
 return x
def oracle(name):
 source=subprocess.check_output(['git','show',f'd23da0e:verl/verl/experimental/natural_continuation_boundary_return/{name}.py'],cwd=R.parent,text=True)
 p=out/f'oracle_{name}.py.txt';p.write_text(source);m=types.ModuleType('frozen_natural_'+name);m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(source,str(p),'exec'),m.__dict__);return m
runtime=oracle('runtime');adapter=oracle('reward_adapter');scores={};summary={}
for mode in ['off','shadow','replace']:
 cfg=BoundaryReturnConfig(mode=mode,long_response_length=8192,max_concurrent_requests=4,request_batch_size=8,long_reward_chunk_size=3,seed=42)
 kw=dict(config=cfg,policy_version=0,sampling_params=dict(temperature=1.,top_p=1.,top_k=-1,ignore_eos=False),eos_token_id=tok.eos_token_id,short_response_length=2048,max_model_len=9216)
 client=Client();result=NCBRHook().apply(batch,raw,raw_verifier_extras=extras,continuation_client=client,score_long=long_score,**kw)
 work=copy.deepcopy(batch);oc=Client()
 if mode!='off':
  cap=runtime.run_boundary_continuations(rollout_batch=work,client=oc,**kw)
  assert [dataclasses.asdict(r) for r in cap.requests]==[dataclasses.asdict(r) for r in actual.aux['capture'].requests]
  adapted=adapter.apply_boundary_return(work,capture=cap,long_reward_output=long_score(work,cap.generations,cfg),config=cfg)
  assert np.array_equal(adapted.task_score_delta,actual.aux['result'].task_score_delta,equal_nan=True)
 assert torch.equal(result.effective_scores,work.batch['token_level_scores']);assert client.events==oc.events
 hb=copy.deepcopy(batch)
 if mode=='replace':
  assert torch.equal(result.effective_scores,actual.effective_scores)
  hb.non_tensor_batch['acc']=result.aux['result'].boundary_acc;work.non_tensor_batch['acc']=adapted.boundary_acc
 assert filter_dapo_generation_batch(hb,'acc').non_tensor_batch['trajectory_id'].tolist()==filter_dapo_generation_batch(work,'acc').non_tensor_batch['trajectory_id'].tolist()
 delta=result.effective_scores-raw;mask=batch.batch['response_mask'];allowed=torch.zeros_like(delta,dtype=torch.bool);allowed[torch.arange(len(batch)),mask.sum(-1).long()-1]=True
 assert torch.equal(delta[~allowed],torch.zeros_like(delta[~allowed]));assert torch.equal(batch.batch['token_level_scores'],raw)
 scores[mode]=result.effective_scores;summary[mode]=dict(exact=True,changed_rows=int((delta!=0).any(-1).sum()),calls=len(client.events))
actor=torch.load(sorted(Path(record['hook_output']).parent.glob('*trainer_actor_batch.pt'))[0],weights_only=False)
adv,_=compute_grpo_outcome_advantage(scores['replace'],batch.batch['response_mask'],batch.non_tensor_batch['uid'])
lookup={str(t):i for i,t in enumerate(batch.non_tensor_batch['trajectory_id'])}
for j,t in enumerate(actor.non_tensor_batch['trajectory_id']):
 i=lookup[str(t)];assert torch.equal(actor.batch['responses'][j],batch.batch['responses'][i]);assert torch.equal(actor.batch['advantages'][j],adv[i]);assert torch.equal(actor.batch['token_level_scores'][j],scores['replace'][i])
torch.save(dict(batch=batch,raw=raw,extras=extras,scores=scores),out/'actor_input.pt')
(out/'long_verifier.json').write_text(json.dumps(long_evidence));(out/'summary.json').write_text(json.dumps(dict(status='PASS',modes=summary,actual_actor_rows=len(actor),actual_actor_advantages_exact=True,source_record=record,scope='Real version0 service tokens and verifier scalars, independently reverified; original adapter and hook exact; selected natural event, not population sampling'),indent=2))
print((out/'summary.json').read_text())
