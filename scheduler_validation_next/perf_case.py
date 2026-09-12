"""One fresh eight-replica continuation timing instance using the production client."""
import argparse, asyncio, dataclasses, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'verl'))
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--scheduler',choices=['fixed_wave','work_conserving'],required=True);p.add_argument('--concurrency',type=int,required=True);p.add_argument('--count',type=int,default=256);p.add_argument('--natural',action='store_true');a=p.parse_args()
OUT=ROOT/'evidence'/a.case
OUT.mkdir(exist_ok=False)
import ray, torch
from omegaconf import OmegaConf
from verl.workers.rollout.llm_server import LLMServerClient, GlobalRequestLoadBalancer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
from verl.workers.rollout.replica import RolloutMode
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations

class Server(vLLMHttpServer):
    def __init__(self, config, model, rank):
        super().__init__(config, model, RolloutMode.STANDALONE, [], rank, 0, 1, 1,
                         os.environ['CUDA_VISIBLE_DEVICES'])
    def receipt(self):
        import vllm,verl,ray,torch
        return dict(pid=os.getpid(),cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'],modules={m.__name__:m.__file__ for m in [vllm,verl,ray,torch]},tp=1,node_id=ray.get_runtime_context().get_node_id())

async def main():
    source=Path('/tmp/ncbr_8gpu_scale_20260911/validation/evidence')
    cfg=OmegaConf.load(source/'train_dapo_vanilla_replace_v6/config.yaml')
    cfg.actor_rollout_ref.rollout.load_format='auto'
    # Same engine configuration in every arm, including cache and determinism settings.
    (OUT/'service_config.yaml').write_text(OmegaConf.to_yaml(cfg.actor_rollout_ref.rollout))
    from workload import make_batch,validate,sha,replay_params
    rows=json.loads((ROOT/'workload.json').read_text())[:a.count]
    assert len(rows)==a.count;validate(rows)
    batch=make_batch(rows)
    by_id={r['request_id']:r for r in rows}
    effective=[]
    from transformers import AutoTokenizer
    eos_token_id=AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path,local_files_only=True).eos_token_id
    begin=time.perf_counter()
    ray.init(address='local',_temp_dir=os.environ['RAY_TMPDIR'],namespace=a.case,
             num_cpus=24,include_dashboard=False,log_to_driver=True)
    servers={str(i):ray.remote(num_gpus=1,num_cpus=1)(Server).remote(cfg.actor_rollout_ref.rollout,cfg.actor_rollout_ref.model,i) for i in range(8)}
    await asyncio.gather(*(s.launch_server.remote() for s in servers.values()))
    await asyncio.gather(*(s.set_global_steps.remote(0) for s in servers.values()))
    lb=GlobalRequestLoadBalancer.remote(servers)
    assert (await lb.get_status.remote())["active_servers"] == 8
    startup=time.perf_counter()-begin
    (OUT/'servers.json').write_text(json.dumps(await asyncio.gather(*(s.receipt.remote() for s in servers.values()))))
    # Identical per-service warmup, outside the timed continuation and routing cache.
    warm=time.perf_counter()
    for s in servers.values():
        await s.generate_grouped.remote(prompt_ids=[1,2,3,4],sampling_params=dict(n=1,max_tokens=8,temperature=1.,top_p=1.,top_k=-1,seed=42,ignore_eos=False),request_id='warmup')
        await s.wait_for_requests_to_drain.remote()
    warmup=time.perf_counter()-warm
    class Events(list):
        def append(self,event):
            super().append(event)
            with (OUT/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
    events=Events()
    class Client(LLMServerClient):
        async def start_grouped(self,request_id,**kw):
            r=by_id[request_id]
            assert kw['prompt_ids']==r['input_token_ids']
            assert kw['sampling_params']['max_tokens']==r['max_tokens']
            kw['sampling_params']=replay_params(r,kw['sampling_params'],a.natural)
            effective.append(dict(request_id=request_id,**kw))
            start=time.perf_counter();events.append(dict(event='dispatch',request_id=request_id,time=start))
            tracked=await super().start_grouped(request_id,**kw)
            events.append(dict(event='acquire',request_id=request_id,server_id=tracked.server_id,time=time.perf_counter()))
            original_result=tracked.result
            async def result():
                outputs=await original_result()
                (OUT/f'{request_id}.json').write_text(json.dumps([o.model_dump() for o in outputs]))
                events.append(dict(event='generation_complete',request_id=request_id,time=time.perf_counter()))
                return outputs
            tracked.result=result
            original=tracked.release
            async def release():
                events.append(dict(event='release_start',request_id=request_id,time=time.perf_counter()))
                await original()
                events.append(dict(event='release_ack',request_id=request_id,time=time.perf_counter()))
            tracked.release=release
            return tracked
    client=Client(cfg,lb)
    boundary=BoundaryReturnConfig(mode='shadow',scheduler=a.scheduler,max_concurrent_requests=a.concurrency,
                                  request_batch_size=256,long_reward_chunk_size=3,seed=42)
    begin=time.perf_counter();epoch_start=time.time()
    try:
        capture=await run_boundary_continuations(config=boundary,rollout_batch=batch,client=client,
            eos_token_id=eos_token_id,short_response_length=2048,max_model_len=9216,policy_version=0,
            sampling_params=dict(temperature=1.,top_p=1.,top_k=-1,ignore_eos=False))
        seconds=time.perf_counter()-begin
        assert len(capture.requests)==a.count
        assert [r.request_id for r in capture.requests]==[r['request_id'] for r in rows]
        if not a.natural:
            assert all(len(g.tail_token_ids)==by_id[g.request_id]['replay_decode_length'] for g in capture.generations)
        (OUT/'effective_requests.json').write_text(json.dumps(effective))
        await asyncio.gather(*(s.wait_for_requests_to_drain.remote() for s in servers.values()))
        status=await lb.get_status.remote()
        assert status['total_inflight']==0, status
        (OUT/'cleanup.json').write_text(json.dumps(status))
        (OUT/'capture.json').write_text(json.dumps(dict(requests=[dataclasses.asdict(r) for r in capture.requests],
            generations=[dataclasses.asdict(g) for g in capture.generations],intervals=[dataclasses.asdict(i) for i in capture.profiling_intervals])))
        (OUT/'result.json').write_text(json.dumps(dict(status='PASS',scheduler=a.scheduler,concurrency=a.concurrency,
            startup_seconds=startup,warmup_seconds=warmup,continuation_seconds=seconds,epoch_start=epoch_start,epoch_end=epoch_start+seconds,monotonic_start=begin,request_count=a.count,natural=a.natural,workload_sha256=sha(ROOT/'workload.json'),prefill_tokens=sum(len(r['input_token_ids']) for r in rows),batch_size=256,
            tokens=sum(len(g.tail_token_ids) for g in capture.generations),cleanup='drain_and_release_ack'),indent=2))
    finally:
        (OUT/'events.json').write_text(json.dumps(events))
        ray.shutdown()

if __name__=='__main__':asyncio.run(main())
