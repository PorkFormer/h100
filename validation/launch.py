import argparse,json,os,subprocess,sys,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--h',type=int,default=128);p.add_argument('--l',type=int,default=512);p.add_argument('--count',type=int,default=8);p.add_argument('--offset',type=int,default=0);a=p.parse_args()
root=Path(__file__).resolve().parent;out=root/'evidence';case=f'h{a.h}_b{a.offset:04d}'
subprocess.run([sys.executable,str(root/'gate.py')],check=True,stdout=(out/f'{case}_gate.log').open('x'))
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines();uuids=[r.split(',')[1].strip() for r in rows];assert len(uuids)==8
ps=[];start=time.monotonic()
for i,uuid in enumerate(uuids):
    cache=root/f'cache_{case}_{i}';cache.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,NCBR_UUID_COMPAT='1',PYTHONPATH=str(root/'compat')+':'+str(root.parent/'verl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',XDG_CACHE_HOME=str(cache),HF_HOME=str(cache/'hf'),VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TMPDIR=str(cache),VLLM_WORKER_MULTIPROC_METHOD='spawn')
    f=(out/f'{case}_replica{i}.log').open('x');cmd=[sys.executable,str(root/'infer.py'),'--replica',str(i),'--h',str(a.h),'--l',str(a.l),'--count',str(a.count),'--offset',str(a.offset)]
    ps.append((subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True),f,cmd))
peak=[0]*8;timed_out=False
while any(p.poll() is None for p,_,_ in ps):
    if time.monotonic()-start>1800:
        timed_out=True
        for p,_,_ in ps:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        break
    vals=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines();peak=[max(x,int(y)) for x,y in zip(peak,vals,strict=True)];time.sleep(2)
codes=[]
for p,f,_ in ps:
    try:codes.append(p.wait(timeout=30))
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);codes.append(p.wait())
    f.close()
(out/f'{case}_processes.json').write_text(json.dumps(dict(commands=[c for _,_,c in ps],pids=[p.pid for p,_,_ in ps],codes=codes,uuids=uuids,seconds=time.monotonic()-start,peak_sampled_mib=peak,timed_out=timed_out),indent=2))
sys.exit(0 if codes==[0]*8 else 1)
