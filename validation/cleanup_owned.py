"""Inspect or terminate only processes carrying this run's private audit marker."""
import argparse,json,os,time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args();owned=[];receipt=[]
for proc in psutil.process_iter(['pid','cmdline','create_time']):
    if proc.pid==os.getpid():continue
    try:
        marker=proc.environ().get('NCBR_AUDIT_DIR','')
        if marker.startswith(str(root/'evidence')+'/'):
            owned.append(proc);receipt.append(dict(pid=proc.pid,command=proc.info['cmdline'],create_time=proc.info['create_time'],marker=marker,status=proc.status()))
    except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):pass
result=dict(apply=a.apply,owned=receipt)
if a.apply:
    for proc in owned:
        try:proc.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=8)
    for proc in alive:
        try:proc.kill()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(alive,timeout=5)
    result['remaining']=[p.pid for p in alive if p.status()!=psutil.STATUS_ZOMBIE]
(root/'evidence'/f'cleanup_owned_{time.time_ns()}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
