"""Actual trainer fault barriers, using fixed smoke budget and real continuation services."""
import argparse,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
summary=json.loads((root/'evidence/fault_summary.json').read_text()) if a.resume else []
faults=['missing','duplicate','version','verifier_error','verifier_timeout','release']
(root/'evidence/fault_manifest.json').write_text(json.dumps(dict(faults=faults,H=128,L=512,prompts='first two frozen manifest prompts',entry='standard GRPO GSPO replace',seed=42,scope='Controlled service/callback errors; never represented as natural verifier transitions'),indent=2))
for fault in faults:
    if any(r['fault']==fault and r['status']=='PASS' for r in summary):continue
    cmd=[sys.executable,str(root/'launch_trainer.py'),'--entry','standard','--loss','gspo','--mode','replace','--fault',fault]
    result=subprocess.run(cmd)
    case=root/'evidence'/f'train_standard_gspo_replace_fault_{fault}_v6';audit=case/'worker_audit'
    if not (case/'process.json').exists():
        raise RuntimeError(f'{fault}: launch gate failed before trainer; preserve evidence before retry')
    assert not json.loads((case/'process.json').read_text()).get('timed_out'),case
    injected=[json.loads(p.read_text()) for p in audit.glob('*_fault.json')];steps=[];published=[];sleeps=[]
    for p in audit.glob('*.json'):
        value=json.loads(p.read_text())
        if value.get('event')=='actual_AdamW_step':steps.append(value)
        if p.name.endswith('_publish_done.json'):published.append(value['version'])
        if value.get('event')=='sleep_attempt':sleeps.append(value['time_ns'])
    if injected:
        assert result.returncode!=0 and not steps and all(v==0 for v in published),(fault,result.returncode,len(steps),published)
        if fault=='release':assert not any(t>=min(x['time_ns'] for x in injected) for t in sleeps)
        status='PASS'
    else:status='UNCOVERED' if result.returncode==0 else 'FAIL_BEFORE_INJECTION'
    summary.append(dict(fault=fault,status=status,process_code=result.returncode,injections=len(injected),optimizer_receipts=len(steps),published_versions=sorted(set(published)),sleep_attempts=len(sleeps)))
    (root/'evidence/fault_summary.json').write_text(json.dumps(summary,indent=2))
    if status=='FAIL_BEFORE_INJECTION':sys.exit(1)
