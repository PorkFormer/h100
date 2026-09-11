import json,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent/'verl'))
case=root/'evidence/train_standard_gspo_replace_ref_v6';audit=case/'worker_audit';by_pid={}
for p in sorted(audit.glob('*forward_meta.json')):
    meta=json.loads(p.read_text())
    if meta['forward_only']:
        by_pid.setdefault(p.name.split('_')[0],{}).setdefault(meta['optimizer_present'],[]).append(p)
def exact(a,b):
    if a.dtype!=b.dtype:return False
    if getattr(a,'is_nested',False):
        if not getattr(b,'is_nested',False):return False
        x,y=a.unbind(),b.unbind()
        return len(x)==len(y) and all(torch.equal(u,v) for u,v in zip(x,y,strict=True))
    return torch.equal(a,b)
checks=[]
for pid,roles in by_pid.items():
    if False not in roles:continue
    assert len(roles[False])==len(roles[True])==1
    actor=torch.load(roles[True][0].with_name(roles[True][0].name.replace('_meta.json','_input.pt')),weights_only=False)
    ref=torch.load(roles[False][0].with_name(roles[False][0].name.replace('_meta.json','_input.pt')),weights_only=False)
    keys=[]
    for key in ['input_ids','position_ids','attention_mask','loss_mask','responses']:
        if key in actor:
            assert key in ref and exact(actor[key],ref[key]),(pid,key)
            keys.append(key)
    assert 'input_ids' in keys and 'loss_mask' in keys
    checks.append(dict(pid=pid,exact_input_keys=keys))
assert len(checks)==8,len(checks)
batches=[]
for p in audit.glob('*trainer_actor_batch.pt'):
    b=torch.load(p,weights_only=False);ref=b.batch['ref_log_prob'];old=b.batch['old_log_probs'];mask=b.batch['response_mask']
    assert ref.shape==old.shape==mask.shape and ref.shape[-1]<=2048
    assert torch.isfinite(ref).all()
    batches.append(dict(shape=list(ref.shape),old_ref_exact=torch.equal(old,ref)))
assert len(batches)==1
(root/'evidence/reference_alignment.json').write_text(json.dumps(dict(status='PASS',workers=checks,actor_batches=batches),indent=2))
