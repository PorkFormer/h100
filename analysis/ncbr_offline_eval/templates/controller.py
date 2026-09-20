"""Manually selected checkpoint, independent horizons, eight reserved GPUs."""
import fcntl, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
from common import ROOT, atomic_json, sha, canonical, check_rows, prompts, models, verify_frozen
def read(p): return json.loads(Path(p).read_text())
def status(stage, **kw): atomic_json(ROOT/'status.json',dict(stage=stage,time=time.time(),**kw))
CONFIG=read(ROOT/'run_config.json')
NODE=CONFIG['node']
CACHE=CONFIG['cache_root']

def snapshot():
    import ray
    nid=ray.get_runtime_context().get_node_id()
    assert next(n for n in ray.nodes() if n['NodeID']==nid)['NodeManagerAddress']==NODE
    def cmd(args): return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=20).strip()
    return dict(time=time.time(),node_id=nid,gpus=cmd(['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),apps=cmd(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']),available=ray._private.state.available_resources_per_node().get(nid,{}))
def idle(s):
    rows=[x.split(',') for x in s['gpus'].splitlines()]
    return not s['apps'] and s['available'].get('GPU',0)==8 and len(rows)==8 and all(float(r[2])<=1024 and float(r[3])<=5 for r in rows)
def validate_shard(path,h,shard):
    with path.open() as stream: rows=[json.loads(line) for line in stream]
    check_rows(rows,h,models(),prompts())
    expected=[(p['prompt_id'],i) for order,p in enumerate(prompts()) for i in range(32) if (order*32+i)%8==shard]
    assert [(r['prompt_id'],r['rollout_index']) for r in rows]==expected,'shard order/coverage'
    assert all(r['worker_shard']==shard for r in rows)
    return rows

def merge(h, successes):
    assert set(successes)==set(range(8))
    rows=[]
    for s in range(8): rows+=validate_shard(Path(successes[s]),h,s)
    check_rows(rows,h,models(),prompts(),complete=True); assert len(rows)==3200
    dest=ROOT/f'formal/generation/h{h}/raw_generations.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    atomic_json(dest.parent/'generation_receipt.json',dict(passed=True,rows=len(rows),selected_shards=successes,sha256=sha(dest)))

OWNED=[]
def run_reserved():
    try:return _run_reserved()
    finally:
        for p in OWNED:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()

def _run_reserved():
    import ray
    assert len(ray.get_gpu_ids())==8
    initial=snapshot(); assert not initial['apps'],'collision after reservation'
    mapping={int(r.split(',')[0]):r.split(',')[1].strip() for r in initial['gpus'].splitlines()}
    assigned=sorted(int(float(x)) for x in ray.get_gpu_ids());assert assigned==sorted(mapping) and len(mapping)==8
    atomic_json(ROOT/'reservation.json',dict(node_id=initial['node_id'],gpu_ids=assigned,uuids=mapping,time=time.time()))
    verify_frozen(full_weights=True)
    for h in (2048,8192):
        success={};attempts={s:0 for s in range(8)};running={};last=0;started=time.time()
        while len(success)<8:
            for s in range(8):
                if s in success or s in running:continue
                if attempts[s]>=3:continue
                # A failed worker may have left GPU children; wait without killing unrelated work.
                apps=snapshot()['apps']
                if any(line.split(',')[0].strip()==mapping[s] for line in apps.splitlines()):continue
                attempts[s]+=1;a=attempts[s]
                d=ROOT/f'formal/attempts/h{h}/shard{s}/attempt{a}';d.mkdir(parents=True,exist_ok=False)
                cache=Path(CACHE)/f'h{h}s{s}a{a}';cache.mkdir(parents=True,exist_ok=True)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(s),'EXPECTED_GPU_UUID':mapping[s], 'VLLM_WORKER_MULTIPROC_METHOD':'spawn','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VLLM_PORT':str(CONFIG['port_base']+s*128),'VLLM_RPC_BASE_PATH':str(cache),'TMPDIR':str(cache),'PYTHONPATH':str(ROOT)}
                for key,sub in [('XDG_CACHE_HOME','xdg'),('PYTHONPYCACHEPREFIX','py'),('TORCHINDUCTOR_CACHE_DIR','ti'),('TRITON_CACHE_DIR','tr'),('FLASHINFER_WORKSPACE_BASE','fi'),('TORCH_EXTENSIONS_DIR','te')]:env[key]=str(cache/sub)
                log=(d/'worker.log').open('x')
                p=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'--model-id',models()[0]['model_id'],'--horizon',str(h),'--gpu',str(s),'--shard',str(s),'--attempt-dir',str(d)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                OWNED.append(p)
                running[s]=(p,log,d)
            for s,(p,log,d) in list(running.items()):
                if p.poll() is None:continue
                log.close()
                if p.returncode != 0:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                path=d/f'generation/h{h}/raw_generations.jsonl';error=None
                try:
                    assert p.returncode==0,f'worker exit {p.returncode}'
                    validate_shard(path,h,s);success[s]=str(path)
                except Exception as exc:error=repr(exc)
                atomic_json(d/'receipt.json',dict(returncode=p.returncode,error=error,passed=error is None,time=time.time()))
                del running[s]
            if not running and any(attempts[s]>=3 and s not in success for s in range(8)):
                raise RuntimeError(f'H{h}: exhausted three attempts; successful shards retained')
            if time.time()-last>=300:
                paths=list((ROOT/'formal/attempts').glob('**/raw_generations.jsonl'))
                counts={str(p.relative_to(ROOT)):sum(1 for _ in p.open()) for p in paths}
                try: gpu=snapshot()
                except Exception as exc:gpu={'monitoring_error':repr(exc)}
                record=dict(horizon=h,elapsed_seconds=time.time()-started,successful_shards=list(success),attempts=attempts,generated_by_attempt=counts,gpu=gpu)
                atomic_json(ROOT/f'monitoring/{int(time.time())}.json',record);status('GENERATING',**record);last=time.time()
            time.sleep(5)
        merge(h,success)
    # Workers have exited normally. Wait for their GPU contexts to disappear before releasing reservation.
    while True:
        s=snapshot();atomic_json(ROOT/'gpu_release.json',s)
        if not s['apps']:break
        time.sleep(10)
    return dict(generation_complete=True,worker_exit_codes_zero=True,gpu_processes_released=True)

def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for item in read(ROOT/'freeze.json')['files']:
        assert sha(item['path'])==item['sha256'],'frozen source/input drift: '+item['path']
    atomic_json(ROOT/'dispatch.json',dict(pid=os.getpid(),time=time.time()))
    status('VERIFY_CHECKPOINT');verify_frozen(full_weights=True)
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy,PlacementGroupSchedulingStrategy
    from ray.util.placement_group import placement_group,remove_placement_group
    ray.init(address=CONFIG['ray_address'],namespace=CONFIG['namespace'],logging_level='ERROR',runtime_env={'env_vars':{'PYTHONPATH':str(ROOT),'PYTHONPYCACHEPREFIX':CACHE+'/controller/py'}})
    node=next(n for n in ray.nodes() if n['Alive'] and n['NodeManagerAddress']==NODE)
    probe=ray.remote(num_cpus=0,max_retries=0)(snapshot).options(scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'],soft=False))
    while True:
        s=ray.get(probe.remote());atomic_json(ROOT/'preflight.json',s)
        if idle(s):break
        status('WAIT_NODE',snapshot=s);time.sleep(30)
    pg=placement_group([{'CPU':1,'GPU':8,'node:'+NODE:0.001}],strategy='STRICT_PACK',name=CONFIG['namespace'])
    try:
        status('WAIT_RESERVATION');ray.get(pg.ready())
        task=ray.remote(num_cpus=1,num_gpus=8,max_retries=0)(run_reserved).options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,placement_group_bundle_index=0))
        result=ray.get(task.remote());atomic_json(ROOT/'generation_completion.json',result)
    finally:
        remove_placement_group(pg)
        while ray.util.placement_group_table(pg).get('state')!='REMOVED':time.sleep(2)
        atomic_json(ROOT/'resource_release.json',dict(placement_removed=True,shared_ray_retained=True,time=time.time()))
        ray.shutdown()
    status('SCORING')
    for attempt_no in range(1,4):
        with (ROOT/f'score_attempt{attempt_no}.log').open('x') as log:
            code=subprocess.call([sys.executable,str(ROOT/'results.py')],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        if code==0:break
    else:raise RuntimeError('scoring/verification failed after three attempts; generation retained')
    verify_frozen(full_weights=True)
    status('COMPLETE',**result,scoring_complete=True,placement_removed=True,normal_exit=True,total_rows=6400)
    atomic_json(ROOT/'SHA256SUMS.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS.json','controller.log','controller.lock')})

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('FAILED',error=repr(exc),normal_exit=False);traceback.print_exc();sys.exit(1)
