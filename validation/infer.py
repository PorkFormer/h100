"""Real vLLM token capture. Each fresh UUID-bound process owns one TP=1 replica."""
def main():
    import argparse, dataclasses, json, os, sys, time
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--replica',type=int,required=True);p.add_argument('--h',type=int,default=128);p.add_argument('--l',type=int,default=512);p.add_argument('--count',type=int,default=8);p.add_argument('--offset',type=int,default=0);a=p.parse_args()
    ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
    OUT=ROOT/'evidence'/f'infer_h{a.h}_b{a.offset:04d}_replica{a.replica}';OUT.mkdir(exist_ok=False)
    import torch, numpy as np
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from verl import DataProto
    from verl.experimental.natural_continuation_boundary_return.runtime import build_continuation_requests
    import verl
    (OUT/'provenance.json').write_text(json.dumps(dict(verl=verl.__file__,uuid=str(torch.cuda.get_device_properties(0).uuid),pid=os.getpid())))
    rows=[json.loads(x) for x in (ROOT/'evidence'/'prompts.jsonl').read_text().splitlines()][a.offset:a.offset+a.count][a.replica::8]
    start=time.monotonic()
    llm=LLM(model='/workspace/models/Qwen3-1.7B-Base',dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.35,enforce_eager=True,max_model_len=9216,seed=42,enable_prefix_caching=False,max_num_seqs=4)
    params=SamplingParams(n=4,max_tokens=a.h,temperature=1,top_p=1,top_k=-1,ignore_eos=False,seed=42,output_kind=RequestOutputKind.FINAL_ONLY)
    outputs=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in rows],params,use_tqdm=False)
    flat=[]
    for row,out in zip(rows,outputs,strict=True):
        assert len(out.outputs)==4
        for b in out.outputs:
            flat.append(dict(**row,trajectory_id=f"{row['uid']}:{b.index}",tokens=list(b.token_ids),finish_reason=b.finish_reason,stop_reason=b.stop_reason,policy_version=0))
    (OUT/'short.json').write_text(json.dumps(flat))
    width=max(len(r['prompt_token_ids']) for r in flat);n=len(flat)
    prompts=torch.zeros(n,width,dtype=torch.long);responses=torch.zeros(n,a.h,dtype=torch.long);pm=torch.zeros_like(prompts);rm=torch.zeros_like(responses)
    for i,r in enumerate(flat):
        ids=r['prompt_token_ids'];t=r['tokens'];prompts[i,-len(ids):]=torch.tensor(ids);pm[i,-len(ids):]=1;responses[i,:len(t)]=torch.tensor(t);rm[i,:len(t)]=1
    batch=DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,attention_mask=torch.cat([pm,rm],-1)),non_tensors=dict(uid=np.array([r['uid'] for r in flat],dtype=object),trajectory_id=np.array([r['trajectory_id'] for r in flat],dtype=object),rollout_policy_version=np.array([0]*n,dtype=object),finish_reason=np.array([r['finish_reason'] for r in flat],dtype=object)))
    requests,mask=build_continuation_requests(batch,policy_version=0,short_response_length=a.h,long_response_length=a.l,max_model_len=9216,eos_token_id=llm.get_tokenizer().eos_token_id,base_seed=42)
    (OUT/'requests.json').write_text(json.dumps([dataclasses.asdict(r) for r in requests]))
    for arm in ['repeat_0','repeat_1']:
        saved=[]
        for start_idx in range(0,len(requests),4):
            wave=requests[start_idx:start_idx+4]
            generated=llm.generate([dict(prompt_token_ids=list(r.input_token_ids)) for r in wave],[SamplingParams(n=1,max_tokens=r.max_tokens,temperature=1,top_p=1,top_k=-1,ignore_eos=False,seed=r.seed,output_kind=RequestOutputKind.FINAL_ONLY) for r in wave],use_tqdm=False)
            for req,out in zip(wave,generated,strict=True):
                assert len(out.outputs)==1
                b=out.outputs[0];saved.append(dict(request_id=req.request_id,tokens=list(b.token_ids),finish_reason=b.finish_reason,stop_reason=b.stop_reason))
        (OUT/f'{arm}_continuations.json').write_text(json.dumps(saved))
    (OUT/'summary.json').write_text(json.dumps(dict(status='CAPTURED',seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),short_rows=n,cap_hits=int(mask.sum()),scope='Direct vLLM capture; runtime source equivalence and remote release not yet verified')))

if __name__ == '__main__':
    main()
