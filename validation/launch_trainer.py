import argparse,json,os,subprocess,sys,time,signal
from pathlib import Path
root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--mode',required=True);p.add_argument('--loss',default='vanilla');p.add_argument('--ref',action='store_true');p.add_argument('--live-oracle-smoke',action='store_true');p.add_argument('--live-oracle-full',action='store_true');p.add_argument('--fault',default=None,choices=['missing','duplicate','version','verifier_error','verifier_timeout','release']);a=p.parse_args();case=root/'evidence'/f'train_{a.entry}_{a.loss}_{a.mode}{"_ref" if a.ref else ""}{("_fault_"+a.fault) if a.fault else ""}{"_live_oracle_smoke" if a.live_oracle_smoke else ""}{"_live_oracle_full" if a.live_oracle_full else ""}_v6';case.mkdir(exist_ok=True)
idle_samples=[];idle_streak=0
for attempt in range(31):
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle_samples.append(values)
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    idle_streak=idle_streak+1 if not compute and all(int(x.strip())==0 for row in values.splitlines() for x in row.split(',')) else 0
    if idle_streak>=2:break
    if attempt==30:raise RuntimeError('GPUs did not become idle before case launch')
    time.sleep(2)
(case/'prelaunch_idle.json').write_text(json.dumps(idle_samples))
with (case/'gate.log').open('x') as f:subprocess.run([sys.executable,str(root/'gate.py')],check=True,stdout=f)
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();cache=case/'cache';cache.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),NCBR_UUID_COMPAT='1',NCBR_AUDIT_DIR=str(case/'worker_audit'),PYTHONPATH=str(root/'compat')+':'+str(root.parent/'verl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',XDG_CACHE_HOME=str(cache),HF_HOME=str(cache/'hf'),VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TMPDIR=str(root.parent/'t'/__import__('hashlib').sha256(case.name.encode()).hexdigest()[:6]),RAY_TMPDIR=str(root.parent/'r'/__import__('hashlib').sha256(case.name.encode()).hexdigest()[:6]),RAY_ADDRESS='local',VLLM_WORKER_MULTIPROC_METHOD='spawn')
if a.fault:env['NCBR_FAULT']=a.fault
if a.live_oracle_smoke or a.live_oracle_full:env['NCBR_LIVE_ORACLE']='1'
Path(env['TMPDIR']).mkdir(parents=True,exist_ok=False)
cmd=[sys.executable,str(root/'trainer_case.py'),'--entry',a.entry,'--mode',a.mode,'--loss',a.loss]+(['--ref'] if a.ref else [])+(['--fault',a.fault] if a.fault else [])+(['--live-oracle-smoke'] if a.live_oracle_smoke else [])+(['--live-oracle-full'] if a.live_oracle_full else [])
start=time.monotonic();timed_out=False
with (case/'run.log').open('x') as f:
    proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=proc.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
        try:code=proc.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
(case/'process.json').write_text(json.dumps(dict(command=cmd,pid=proc.pid,code=code,seconds=time.monotonic()-start,timed_out=timed_out),indent=2));sys.exit(code)
