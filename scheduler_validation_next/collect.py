"""Bounded frozen-weight NCBR execution, no trainer or optimizer."""
import argparse,asyncio,dataclasses,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();OUT=ROOT/'evidence'/a.case;OUT.mkdir(exist_ok=False)
import ray
from omegaconf import OmegaConf
from verl.workers.rollout.llm_server import LLMServerClient,GlobalRequestLoadBalancer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
from verl.workers.rollout.replica import RolloutMode
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations
from workload import make_batch,validate,sha
class Server(vLLMHttpServer):
    def __init__(self,config,model,rank):super().__init__(config,model,RolloutMode.STANDALONE,[],rank,0,1,1,os.environ['CUDA_VISIBLE_DEVICES'])
    def receipt(self):return dict(pid=os.getpid(),cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'])
async def main():
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    source=Path('/tmp/ncbr_8gpu_scale_20260911/validation/evidence')
    cfg=OmegaConf.load(source/'train_dapo_vanilla_replace_v6/config.yaml');cfg.actor_rollout_ref.rollout.load_format='auto'
    (OUT/'service_config.yaml').write_text(OmegaConf.to_yaml(cfg.actor_rollout_ref.rollout))
    eos=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path,local_files_only=True).eos_token_id
    prompts=[json.loads(x) for x in (ROOT/'prompts.jsonl').read_text().splitlines()]
    ray.init(address='local',_temp_dir=os.environ['RAY_TMPDIR'],namespace=a.case,num_cpus=24,include_dashboard=False)
    servers={str(i):ray.remote(num_gpus=1,num_cpus=1)(Server).remote(cfg.actor_rollout_ref.rollout,cfg.actor_rollout_ref.model,i) for i in range(8)}
    try:
        await asyncio.gather(*(s.launch_server.remote() for s in servers.values()))
        await asyncio.gather(*(s.set_global_steps.remote(0) for s in servers.values()))
        (OUT/'servers.json').write_text(json.dumps(await asyncio.gather(*(s.receipt.remote() for s in servers.values()))))
        lb=GlobalRequestLoadBalancer.remote(servers);assert (await lb.get_status.remote())['active_servers']==8
        mappings={}
        class Client(LLMServerClient):
            async def start_grouped(self,request_id,**kw):
                h=await super().start_grouped(request_id,**kw);mappings[request_id]=h.server_id;return h
        client=Client(cfg,lb);frozen=[]
        for start in range(0,len(prompts),128):
            rows=prompts[start:start+128]
            async def short(i,r):
                outputs=await servers[str(i%8)].generate_grouped.remote(prompt_ids=r['prompt_token_ids'],sampling_params=dict(n=4,max_tokens=2048,temperature=1.,top_p=1.,top_k=-1,seed=42,ignore_eos=False,output_kind=RequestOutputKind.FINAL_ONLY),request_id=f'short-{start+i}')
                assert len(outputs)==4
                return [dict(uid=r['uid'],prompt_token_ids=r['prompt_token_ids'],trajectory_id=f"{r['uid']}:{b}",tokens=o.token_ids,finish_reason=o.extra_fields['finish_reason'],original_prompt_index=start+i,original_engine=str(i%8)) for b,o in enumerate(outputs)]
            groups=await asyncio.gather(*(short(i,r) for i,r in enumerate(rows)))
            flat=[x for g in groups for x in g]
            (OUT/f'short_{start:04d}.json').write_text(json.dumps(flat))
            hits=[r for r in flat if r['finish_reason']=='length' and len(r['tokens'])==2048 and r['tokens'][-1]!=eos]
            if hits:
                batch=make_batch(hits)
                cap=await run_boundary_continuations(config=BoundaryReturnConfig(mode='shadow',scheduler='work_conserving',max_concurrent_requests=64,request_batch_size=256,seed=42),rollout_batch=batch,client=client,eos_token_id=eos,short_response_length=2048,max_model_len=9216,policy_version=0,sampling_params=dict(temperature=1.,top_p=1.,top_k=-1,ignore_eos=False))
                saved=dict(requests=[dataclasses.asdict(r) for r in cap.requests],generations=[dataclasses.asdict(g) for g in cap.generations])
                (OUT/f'capture_{start:04d}.json').write_text(json.dumps(saved))
                for r,g in zip(saved['requests'],saved['generations'],strict=True):
                    assert r['request_id']==g['request_id']
                    frozen.append(dict(**r,replay_decode_length=len(g['tail_token_ids']),natural_tail_token_ids=g['tail_token_ids'],observed_natural_decoded_length=len(g['tail_token_ids']),finish_reason=g['finish_reason'],source_batch=f'{a.case}/short_{start:04d}.json',source_step=0,original_engine_mapping=mappings[r['request_id']]))
            (OUT/'progress.json').write_text(json.dumps(dict(prompts_completed=start+len(rows),short_trajectories=(start+len(rows))*4,unique_continuations=len({r['request_id'] for r in frozen}))))
            print('COLLECTION_PROGRESS',start+len(rows),len(frozen),flush=True)
            if len(frozen)>=256:break
        assert len(frozen)>=256,'Insufficient cap hits in predeclared prompt pool'
        # JSON canonicalization converts dataclass tuples to lists before validation.
        frozen=json.loads(json.dumps(frozen));validate(frozen)
        (OUT/'all_requests.json').write_text(json.dumps(frozen))
        selected=frozen[:256];validate(selected)
        with (ROOT/'workload.json').open('x') as f:json.dump(selected,f)
        (ROOT/'workload_manifest.json').write_text(json.dumps(dict(count=256,unique_count=len({r['request_id'] for r in selected}),sha256=sha(ROOT/'workload.json'),all_collected=len(frozen),selection='First 256 in deterministic prompt then branch order; complete raw batches retained',prompts_sha256=sha(ROOT/'prompts.jsonl'),source_case=a.case,model_sha256=json.loads((ROOT/'preflight.json').read_text())['model_sha256']),indent=2))
        await asyncio.gather(*(s.wait_for_requests_to_drain.remote() for s in servers.values()))
        status=await lb.get_status.remote();assert status['total_inflight']==0
        (OUT/'cleanup.json').write_text(json.dumps(status));print('COLLECTION_PASS',flush=True)
    finally:ray.shutdown()
if __name__=='__main__':asyncio.run(main())
