import json
from pathlib import Path
import torch
root=Path(__file__).resolve().parent/'evidence';result={}
for h in [128,2048]:
    count=0
    for rank in range(8):
        for name in ['vanilla','gspo']:
            base=torch.load(root/f'actor_h{h}'/f'rank{rank}_{name}_off_repeat0.pt',weights_only=False)
            for mode in ['shadow','replace']:
                other=torch.load(root/f'actor_h{h}'/f'rank{rank}_{name}_{mode}_repeat0.pt',weights_only=False)
                assert torch.equal(base['gradient_shard'],other['gradient_shard'])
                for x,y in zip(base['inputs'],other['inputs'],strict=True):
                    for key in ['old','log_prob','advantage','mask','loss']:assert torch.equal(x[key],y[key])
                count+=1
    result[h]=dict(status='PASS',exact_cross_mode_comparisons=count,scope='Observed effective rewards are identical across modes in this frozen sample')
(root/'actor_cross_mode.json').write_text(json.dumps(result,indent=2))
