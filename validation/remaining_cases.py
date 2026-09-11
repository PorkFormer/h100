"""Sequential dispatch; stop on unexpected failure, retain DAPO exhaustion as uncovered."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
cases=[('dapo','vanilla',m,False) for m in ['replace']]+[('standard',loss,m,False) for loss in ['vanilla','gspo'] for m in ['off','shadow','replace']]+[('standard','gspo','replace',True)]
summary=[]
for entry,loss,mode,ref in cases:
    cmd=[sys.executable,str(root/'launch_trainer.py'),'--entry',entry,'--loss',loss,'--mode',mode]+(['--ref'] if ref else [])
    result=subprocess.run(cmd)
    case=root/'evidence'/f'train_{entry}_{loss}_{mode}{"_ref" if ref else ""}_v6'
    log=(case/'run.log').read_text() if (case/'run.log').exists() else ''
    exhaustion=entry=='dapo' and ('max_num_gen_batches' in log and ('Generated too many; data may be too difficult.' in log))
    summary.append(dict(case=case.name,code=result.returncode,exhaustion=exhaustion))
    (root/'evidence/remaining_cases.json').write_text(json.dumps(summary,indent=2))
    if result.returncode and not exhaustion:sys.exit(result.returncode)
