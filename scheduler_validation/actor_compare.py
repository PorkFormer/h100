"""Compare full frozen vanilla/GSPO loss inputs and gradient shards across schedulers."""
import json
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import torch
R=Path(__file__).resolve().parent;E=R/'evidence';summary={}
for variant in ['work_conserving_4','work_conserving_8']:
    checked=0
    for loss in ['vanilla','gspo']:
        for mode in ['off','shadow','replace']:
            for rank in range(8):
                for repeat in range(2):
                    name=f'rank{rank}_{loss}_{mode}_repeat{repeat}.pt'
                    a=torch.load(E/'actor_fixed_wave_4'/name,weights_only=False)
                    b=torch.load(E/f'actor_{variant}'/name,weights_only=False)
                    assert torch.equal(a['gradient_shard'],b['gradient_shard']), (variant,name)
                    for x,y in zip(a['inputs'],b['inputs'],strict=True):
                        assert x.keys()==y.keys()
                        for key in x:
                            assert torch.equal(x[key],y[key]) if torch.is_tensor(x[key]) else x[key]==y[key], (name,key)
                    checked+=1
    summary[variant]=dict(status='PASS',exact_loss_input_and_gradient_files=checked)
(E/'actor_scheduler_comparison.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
