"""Continue GPU gates sequentially once the already launched old repeat settles."""
import json, subprocess, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
while not (E/'repeat_old_0_process.json').exists():time.sleep(2)
assert json.loads((E/'repeat_old_0_process.json').read_text())['code']==0
for scheduler,concurrency in [('fixed_wave',4),('work_conserving',4),('work_conserving',8)]:
    for _ in range(30):
        p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        if not p and all(int(v.strip())==0 for row in values.splitlines() for v in row.split(',')):break
        time.sleep(2)
    case=f'actor_{scheduler}_{concurrency}'
    with (E/f'{case}_launch.log').open('x') as log:
        subprocess.run([sys.executable,str(R/'launch.py'),'--kind','actor','--case',case,
                        '--scheduler',scheduler,'--concurrency',str(concurrency)],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(R/'actor_compare.py')],check=True)
subprocess.run([sys.executable,str(R/'suite.py')],check=True)
