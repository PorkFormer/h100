"""Isolated launcher with fresh CUDA gates, process-group ownership and raw receipts."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--scheduler',default='fixed_wave');p.add_argument('--concurrency',type=int,default=4);p.add_argument('--kind',choices=['perf','nccl','trainer','actor'],default='perf');p.add_argument('--entry',default='dapo');p.add_argument('--mode',default='replace');p.add_argument('--loss',default='vanilla');a=p.parse_args()
with (E/f'{a.case}_gate.log').open('x') as f:
    subprocess.run([sys.executable,str(R/'gate.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert uuids==json.loads((R/'manifest.json').read_text())['uuid_order']
cache=R/'cache'/a.case;cache.mkdir(parents=True,exist_ok=False)
tmp=Path('/tmp')/('ncs_'+a.case);tmp.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),NCBR_UUID_COMPAT='1',
    PYTHONPATH=str(R.parent/'validation/compat')+':'+str(R.parent/'verl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',XDG_CACHE_HOME=str(cache),HF_HOME=str(cache/'hf'),
    VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TMPDIR=str(tmp),
    RAY_TMPDIR=str(tmp/'ray'),RAY_ADDRESS='local',VLLM_WORKER_MULTIPROC_METHOD='spawn')
if a.kind=='nccl':
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',str(R/'nccl_probe.py')]
elif a.kind=='actor':
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',NCBR_ACTOR_OUT=str(E/a.case))
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
         str(R/'actor_replay.py'),'--h','2048','--source',str(E/'replay'/f'actor_{a.scheduler}_{a.concurrency}.pt')]
elif a.kind=='trainer':
    case=E/a.case;case.mkdir(exist_ok=False)
    env.update(NCBR_CASE_DIR=str(case),NCBR_AUDIT_DIR=str(case/'worker_audit'))
    cmd=[sys.executable,str(R/'trainer_case.py'),'--scheduler',a.scheduler,'--concurrency',str(a.concurrency),
         '--entry',a.entry,'--mode',a.mode,'--loss',a.loss]
else:
    cmd=[sys.executable,str(R/'perf_case.py'),'--case',a.case,'--scheduler',a.scheduler,'--concurrency',str(a.concurrency)]
# Preserve exact executed code separately from later runner fixes.
import hashlib
snapshots={}
for path in [R/'process_cleanup.py',R/'perf_case.py',R/'trainer_case.py',R/'actor_replay.py',R/'launch.py',
             R.parent/'verl/verl/experimental/natural_continuation_boundary_return/runtime.py',
             R.parent/'verl/verl/experimental/natural_continuation_boundary_return/scheduler.py']:
    text=path.read_bytes();snapshots[str(path)]=hashlib.sha256(text).hexdigest()
(E/f'{a.case}_source_hashes.json').write_text(json.dumps(snapshots,indent=2))
start=time.monotonic();peak=[0]*8;timed_out=False;samples=[]
with (E/f'{a.case}.log').open('x') as log:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while proc.poll() is None:
        values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        used=[int(v) for v in values.splitlines()];peak=[max(x,y) for x,y in zip(peak,used,strict=True)]
        samples.append(dict(seconds=time.monotonic()-start,mib=used))
        if time.monotonic()-start>1800:
            timed_out=True;os.killpg(proc.pid,signal.SIGTERM);break
        time.sleep(2)
    try:code=proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
# Settle only descendants carrying this instance's exclusive runtime marker.
# This is process teardown after the timed case, never remote-cleanup attestation.
from process_cleanup import cleanup_owned
cleanup=cleanup_owned(tmp)
(E/f'{a.case}_owned_cleanup.json').write_text(json.dumps(cleanup,indent=2))
if (cleanup['remaining'] or cleanup['errors']) and code==0:code=1
receipt=dict(command=cmd,pid=proc.pid,code=code,timed_out=timed_out,seconds=time.monotonic()-start,
             peak_sampled_mib=peak,samples=samples,uuids=uuids)
(E/f'{a.case}_process.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='samples'}),flush=True)
sys.exit(0 if code==0 and not timed_out else 1)
