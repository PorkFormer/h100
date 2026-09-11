"""Observed group diversity, natural corrections and serving after nonzero updates."""
import collections,json,sys
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent;E=R/'evidence';sys.path.insert(0,str(R.parent/'verl'))
report=[]
for case in sorted(E.glob('train_*_v6')):
 if not (case/'process.json').exists():continue
 audit=case/'worker_audit';groups=[];served=[];steps=collections.defaultdict(list)
 for p in sorted(audit.glob('*.json')):
  v=json.loads(p.read_text())
  if v.get('event')=='actual_AdamW_step':steps[v['pid']].append(v)
 for p in sorted(audit.glob('*rollout_output.pt')):
  b=torch.load(p,weights_only=False);unique=collections.defaultdict(list)
  served.extend(np.asarray(b.non_tensor_batch.get('global_steps',[])).reshape(-1).tolist())
  for i in range(len(b)):
   w=b.batch['prompts'].shape[-1];key=tuple(b.batch['prompts'][i][b.batch['attention_mask'][i,:w].bool()].tolist())
   unique[key].append(tuple(b.batch['responses'][i][b.batch['response_mask'][i].bool()].tolist()))
  groups.extend(dict(rows=len(v),unique_token_trajectories=len(set(v))) for v in unique.values())
 nonzero_versions=[]
 if steps:
  complete_steps=min(len(v) for v in steps.values()) if len(steps)==8 else 0
  if json.loads((case/'process.json').read_text())['code']==0:assert len(steps)==8 and len({len(v) for v in steps.values()})==1
  for i in range(complete_steps):
   if all(v[i]['gradient_nonzero']>0 and v[i]['before']!=v[i]['after'] for v in steps.values()):nonzero_versions.append(i+1)
 corrections=0;potential=0;transitions=collections.Counter();reward_groups=[]
 for p in sorted(audit.glob('*hook_input.pt')):
  before=torch.load(p,weights_only=False);b=before['batch'];acc=np.asarray(before['extras']['acc']);score=np.asarray(before['extras']['score'])
  for uid in dict.fromkeys(b.non_tensor_batch['uid'].tolist()):
   ix=np.asarray(b.non_tensor_batch['uid'])==uid
   reward_groups.append(dict(uid=str(uid),short_acc=acc[ix].tolist(),task_score=score[ix].tolist(),mixed_short_acc=bool(np.std(acc[ix])>0)))
 reward_source='hook_input' if reward_groups else 'unavailable'
 if not reward_groups:
  paths=list(audit.glob('*candidate_before.pt')) or list(audit.glob('*trainer_actor_batch.pt'))
  for p in paths:
   b=torch.load(p,weights_only=False)
   if 'acc' not in b.non_tensor_batch:continue
   reward_source='candidate_before' if 'candidate_before' in p.name else 'trainer_actor_batch'
   acc=np.asarray(b.non_tensor_batch['acc']);score=np.asarray(b.non_tensor_batch.get('score',acc))
   for uid in dict.fromkeys(b.non_tensor_batch['uid'].tolist()):
    ix=np.asarray(b.non_tensor_batch['uid'])==uid
    reward_groups.append(dict(uid=str(uid),short_acc=acc[ix].tolist(),task_score=score[ix].tolist(),mixed_short_acc=bool(np.std(acc[ix])>0)))
 for p in audit.glob('*hook_output.pt'):
  out=torch.load(p,weights_only=False);corrections+=int(out.corrected_mask.sum()) if out.corrected_mask is not None else 0
  if out.aux:
   v=out.aux['result']
   potential+=sum(bool(hit) and float(delta)!=0 for hit,delta in zip(v.hit_response_cap,v.task_score_delta))
   for hit,a,b in zip(v.hit_response_cap,v.short_acc,v.long_acc):
    if hit:transitions[f'{a}->{b}']+=1
 observed=sorted(set(served));served_changed=sorted(set(observed)&set(nonzero_versions))
 report.append(dict(case=case.name,group_count=len(groups),unique_trajectory_counts=dict(collections.Counter(g['unique_token_trajectories'] for g in groups)),groups=groups,short_reward_group_source=reward_source,short_reward_groups=reward_groups,mixed_short_reward_groups=sum(g['mixed_short_acc'] for g in reward_groups),natural_nonzero_task_deltas=potential,natural_corrected_rows=corrections,cap_transitions=dict(transitions),nonzero_weight_change_versions=nonzero_versions,observed_normal_versions=observed,served_after_nonzero_update_versions=served_changed,nonzero_update_serving='PASS' if served_changed else 'UNCOVERED'))
(E/'scale_coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps([{k:v for k,v in x.items() if k not in ['groups','short_reward_groups']} for x in report],indent=2))
