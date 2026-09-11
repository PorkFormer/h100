"""Sequential cases; retain timeout failures and optionally continue after owned cleanup."""
import argparse,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--continue-timeouts',action='store_true');a=p.parse_args()
cases=[('dapo','vanilla',m,False) for m in ['off','shadow','replace']]+[('standard',loss,m,False) for loss in ['vanilla','gspo'] for m in ['off','shadow','replace']]
summary=json.loads((root/'evidence/remaining_cases.json').read_text()) if a.resume else []
for record in summary:
 p=root/'evidence'/record['case']/'process.json'
 if p.exists():record['timed_out']=json.loads(p.read_text()).get('timed_out',False)
for entry,loss,mode,ref in cases:
 name=f'train_{entry}_{loss}_{mode}{"_ref" if ref else ""}_v6'
 if any(r['case']==name for r in summary):continue
 cmd=[sys.executable,str(root/'launch_trainer.py'),'--entry',entry,'--loss',loss,'--mode',mode]+(['--ref'] if ref else [])
 result=subprocess.run(cmd)
 case=root/'evidence'/name
 log=(case/'run.log').read_text() if (case/'run.log').exists() else ''
 exhaustion=entry=='dapo' and 'Generated too many; data may be too difficult.' in log
 proc=json.loads((case/'process.json').read_text()) if (case/'process.json').exists() else {}
 timed_out=bool(proc.get('timed_out'))
 summary.append(dict(case=name,code=result.returncode,exhaustion=exhaustion,timed_out=timed_out))
 (root/'evidence/remaining_cases.json').write_text(json.dumps(summary,indent=2))
 if timed_out and a.continue_timeouts:
  subprocess.run([sys.executable,str(root/'cleanup_owned.py'),'--apply'],check=True)
 elif result.returncode and not exhaustion:sys.exit(result.returncode)
