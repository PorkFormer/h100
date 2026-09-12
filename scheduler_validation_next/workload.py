"""Benchmark-only frozen trace conversion; production request builder is unchanged."""
import hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make_batch(rows):
    import torch,numpy as np
    from verl import DataProto
    w=max(len(r['prompt_token_ids']) for r in rows);n=len(rows)
    p=torch.zeros(n,w,dtype=torch.long);s=torch.zeros(n,2048,dtype=torch.long);pm=torch.zeros_like(p);sm=torch.zeros_like(s)
    for i,r in enumerate(rows):
        x=r['prompt_token_ids'];y=r.get('prefix_token_ids',r.get('tokens'));assert len(y)==2048
        p[i,-len(x):]=torch.tensor(x);pm[i,-len(x):]=1;s[i]=torch.tensor(y);sm[i]=1
    return DataProto.from_dict(tensors=dict(prompts=p,responses=s,attention_mask=torch.cat([pm,sm],-1)),non_tensors={k:np.array(v,dtype=object) for k,v in dict(uid=[r['uid'] for r in rows],trajectory_id=[r['trajectory_id'] for r in rows],rollout_policy_version=[0]*n,finish_reason=['length']*n).items()})
def validate(rows):
    assert len({r['request_id'] for r in rows})==len(rows)
    for r in rows:
        assert r['input_token_ids']==r['prompt_token_ids']+r['prefix_token_ids']
        assert len(r['prefix_token_ids'])==2048 and r['policy_version']==0
        assert 0<r['replay_decode_length']<=r['max_tokens']
        assert r['replay_decode_length']==len(r['natural_tail_token_ids'])
    return True

def replay_params(row,params,natural=False):
    assert params['max_tokens']==row['max_tokens']
    assert params['seed']==row['seed']
    result=dict(params)
    if not natural:result.update(max_tokens=row['replay_decode_length'],ignore_eos=True)
    return result
