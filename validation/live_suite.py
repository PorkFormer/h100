import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent;summary=[]
for horizon in ['smoke','full']:
    cmd=[sys.executable,str(root/'launch_trainer.py'),'--entry','standard','--loss','gspo','--mode','replace',f'--live-oracle-{horizon}']
    result=subprocess.run(cmd)
    case=root/'evidence'/f'train_standard_gspo_replace_live_oracle_{horizon}_v6';audit=case/'worker_audit'
    assert not json.loads((case/'process.json').read_text()).get('timed_out'),case
    complete=list(audit.glob('*live_validation_complete.json'));compare=list(audit.glob('*live_comparison.json'))
    assert result.returncode!=0 and len(complete)==len(compare)==1,(horizon,result.returncode,complete)
    record=json.loads(compare[0].read_text());assert record['request_count']>0
    steps=[json.loads(p.read_text()) for p in audit.glob('*.json') if json.loads(p.read_text()).get('event')=='actual_AdamW_step'];assert not steps
    record.update(horizon=horizon,status='PASS',optimizer_steps=0,expected_diagnostic_stop=True)
    summary.append(record);(root/'evidence/live_oracle_summary.json').write_text(json.dumps(summary,indent=2))
