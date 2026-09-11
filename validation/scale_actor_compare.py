"""Shadow must equal off; replacement may change advantages and gradients naturally."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--tag',default='',choices=['','_natural']);a=p.parse_args()
E=Path(__file__).resolve().parent/'evidence';result={}
for loss in ['vanilla','gspo']:
 changed_adv=changed_grad=0
 for rank in range(8):
  prefix=E/f'actor_h2048{a.tag}'/f'rank{rank}_{loss}'
  base=torch.load(str(prefix)+'_off_repeat0.pt',weights_only=False)
  shadow=torch.load(str(prefix)+'_shadow_repeat0.pt',weights_only=False)
  replace=torch.load(str(prefix)+'_replace_repeat0.pt',weights_only=False)
  assert torch.equal(base['gradient_shard'],shadow['gradient_shard'])
  for x,y,z in zip(base['inputs'],shadow['inputs'],replace['inputs'],strict=True):
   for key in ['old','log_prob','advantage','mask','loss']:assert torch.equal(x[key],y[key]),key
   for key in ['old','log_prob','mask']:assert torch.equal(x[key],z[key]),key
   changed_adv+=int(not torch.equal(x['advantage'],z['advantage']))
  changed_grad+=int(not torch.equal(base['gradient_shard'],replace['gradient_shard']))
 if changed_adv:assert changed_grad>0,loss
 result[loss]=dict(status='PASS',shadow_exact=True,replace_changed_advantage_rows=changed_adv,replace_changed_gradient_shards=changed_grad,natural_gradient_effect='COVERED' if changed_adv and changed_grad else 'UNCOVERED')
(E/f'scale_actor_comparison{a.tag}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
