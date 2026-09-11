"""Separate complete four-update training from timeout and unobserved serving."""
import collections,json,sys
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'verl'));E=R/'evidence';reports=[]
for concurrency in [4,8]:
 for entry,loss,mode in [('dapo','vanilla','shadow'),('dapo','vanilla','replace'),('standard','gspo','replace')]:
  case=f'train_{entry}_{loss}_{mode}_c{concurrency}';folder=E/case;process=E/f'{case}_process.json'
  if not process.exists():reports.append(dict(case=case,status='UNCOVERED'));continue
  proc=json.loads(process.read_text());audit=folder/'worker_audit';steps=collections.defaultdict(list);published=[];served=[];requests=0;release_acks=0;prefix_rows=0;changed_rows=0;known={}
  for path in sorted(audit.glob('*.json')):
   value=json.loads(path.read_text())
   if value.get('event')=='actual_AdamW_step':steps[value['pid']].append(value)
   if value.get('event')=='publish_done':published.append(value['version'])
  for path in audit.glob('*rollout_output.pt'):
   batch=torch.load(path,weights_only=False)
   served.extend(np.asarray(batch.non_tensor_batch.get('global_steps',[])).reshape(-1).tolist())
  for path in audit.glob('*hook_input.pt'):
   record=torch.load(path,weights_only=False);batch=record['batch']
   for i,identity in enumerate(zip(batch.non_tensor_batch['uid'],batch.non_tensor_batch['trajectory_id'],strict=True)):
    known[identity]=batch.batch['responses'][i][batch.batch['response_mask'][i].bool()]
  for path in audit.glob('*hook_output.pt'):
   result=torch.load(path,weights_only=False);capture=result.aux['capture'];requests+=len(capture.requests)
   release_acks+=sum(i.name=='continuation_cleanup_release' for i in capture.profiling_intervals)
   before=torch.load(path.with_name(path.name.replace('_hook_output','_hook_input')),weights_only=False)
   delta=result.effective_scores-before['raw_scores'];mask=before['batch'].batch['response_mask']
   allowed=torch.zeros_like(delta,dtype=torch.bool);allowed[torch.arange(len(mask)),mask.sum(-1).long()-1]=True
   assert not torch.count_nonzero(delta[~allowed]);changed_rows+=int((delta!=0).any(-1).sum())
  for path in audit.glob('*trainer_actor_batch.pt'):
   batch=torch.load(path,weights_only=False);assert batch.batch['responses'].shape[-1]==2048
   for i,identity in enumerate(zip(batch.non_tensor_batch['uid'],batch.non_tensor_batch['trajectory_id'],strict=True)):
    assert torch.equal(known[identity],batch.batch['responses'][i][batch.batch['response_mask'][i].bool()]);prefix_rows+=1
  assert requests==release_acks
  counts=[len(v) for v in steps.values()];logical=min(counts) if len(counts)==8 else 0
  changed_versions=[]
  for i in range(logical):
   if all(v[i]['before']!=v[i]['after'] and v[i]['gradient_nonzero']>0 for v in steps.values()):changed_versions.append(i+1)
  observed=sorted(set(served));published=sorted(set(published))
  success=proc['code']==0 and not proc['timed_out'] and len(counts)==8 and counts==[4]*8 and changed_versions==[1,2,3,4] and all(v in published for v in range(1,5)) and all(v in observed for v in range(1,4))
  reports.append(dict(case=case,status='PASS' if success else 'FAIL_TIMEOUT' if proc['timed_out'] else 'FAIL',seconds=proc['seconds'],logical_updates=logical,nonzero_changed_versions=changed_versions,published=published,actually_served=observed,requests=requests,release_acks=release_acks,prefix_rows_verified=prefix_rows,effective_changed_rows=changed_rows,peak_sampled_mib=proc['peak_sampled_mib']))
(E/'trainer_report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
