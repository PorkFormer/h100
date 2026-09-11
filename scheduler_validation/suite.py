"""Prespecified sequential paired runs; failed instances are never overwritten."""
import json, subprocess, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
def idle():
    for _ in range(30):
        p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        if not p and all(int(v.strip())==0 for row in mem.splitlines() for v in row.split(',')):return
        time.sleep(2)
    raise RuntimeError('Previous instance has not released GPUs; no next launch')
def run(case,scheduler,concurrency,extra=()):
    receipt=E/f'{case}_process.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['code']!=0:raise RuntimeError(f'Existing failure requires review: {case}')
        return
    idle()
    cmd=[sys.executable,str(R/'launch.py'),'--case',case,'--scheduler',scheduler,'--concurrency',str(concurrency),*extra]
    with (E/f'{case}_launch.log').open('x') as log:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    (E/'suite_progress.json').write_text(json.dumps(dict(case=case,code=proc.returncode,time=time.time())))
    if proc.returncode:raise RuntimeError(f'Instance failed: {case}')
for i in range(2):run(f'repeat_old_{i}','fixed_wave',4)
for group,arms in [('scheduler',[('fixed_wave',4),('work_conserving',4)]),
                   ('capacity',[('work_conserving',4),('work_conserving',8)])]:
    for i in range(6):
        for arm in ([0,1] if i%2==0 else [1,0]):
            scheduler,concurrency=arms[arm]
            run(f'{group}_{i}_{"a" if arm==0 else "b"}',scheduler,concurrency)
        subprocess.run([sys.executable,str(R/'performance_report.py')],check=True,stdout=subprocess.DEVNULL)
for concurrency in [4,8]:
    for entry,loss,mode in [('dapo','vanilla','shadow'),('dapo','vanilla','replace'),('standard','gspo','replace')]:
        # Trainer timeout is a result. Continue only after its owned processes settle.
        case=f'train_{entry}_{loss}_{mode}_c{concurrency}'
        try:run(case,'work_conserving',concurrency,['--kind','trainer','--entry',entry,'--loss',loss,'--mode',mode])
        except RuntimeError:
            if not (E/f'{case}_process.json').exists():raise
            idle()
(E/'suite_complete.json').write_text(json.dumps(dict(status='EXECUTED',time=time.time())))
