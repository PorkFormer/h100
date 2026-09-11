import argparse,os,subprocess,sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--h',type=int,default=128);a=p.parse_args()
with (root/'evidence'/f'actor_h{a.h}_v2_gate.log').open('x') as log:subprocess.run([sys.executable,str(root/'gate.py')],stdout=log,check=True)
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',XDG_CACHE_HOME=str(root/'actor_cache'))
cmd=['timeout','1800s',sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',str(root/'actor_replay.py'),'--h',str(a.h)]
start=time.monotonic()
with (root/'evidence'/f'actor_h{a.h}_v2.log').open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
(root/'evidence'/f'actor_h{a.h}_v2_process.json').write_text(json.dumps(dict(command=cmd,code=r.returncode,seconds=time.monotonic()-start)))
sys.exit(r.returncode)
