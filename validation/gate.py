"""Read-only per-device CUDA gate; each probe uses a fresh UUID-bound process."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence'
OUT.mkdir(exist_ok=True)
def run(args, env=None, timeout=90):
    start=time.monotonic()
    try:
        p=subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
        return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, seconds=time.monotonic()-start)
    except subprocess.TimeoutExpired as e:
        return dict(command=args, returncode=None, timeout=True, stdout=str(e.stdout), stderr=str(e.stderr), seconds=time.monotonic()-start)
snapshot=run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
processes=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'])
rows={int(r.split(',')[0]):[v.strip() for v in r.split(',')] for r in snapshot['stdout'].splitlines()}
results=[]
for index in range(8):
    row=rows[index]; uuid=row[1]
    if int(row[2]) != 0 or int(row[3]) != 0 or uuid in processes['stdout']:
        results.append(dict(index=index,uuid=uuid,status='BLOCKED_BUSY')); continue
    code='''import torch, json
print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda}),flush=True)
torch.cuda.set_device(0)
p=torch.cuda.get_device_properties(0)
print(str(p),flush=True)
a=torch.arange(1024,device="cuda",dtype=torch.float32)
assert (a*a).sum().item()==357389824.0
torch.cuda.synchronize()
print("CUDA_COMPUTE_PASS",flush=True)
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
    receipt=run([sys.executable,'-c',code],env)
    results.append(dict(index=index,uuid=uuid,status='PASS' if receipt['returncode']==0 else 'FAIL',receipt=receipt))
report=dict(snapshot=snapshot,processes=processes,devices=results,passed=all(r['status']=='PASS' for r in results))
with (OUT/f'cuda_gate_{time.time_ns()}.json').open('x') as f:
    f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 1)
