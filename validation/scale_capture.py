"""Four prespecified independent blocks; no result-driven sample selection."""
import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
receipts=[]
for offset in [0,128,256,384]:
 cmd=[sys.executable,str(R/'launch.py'),'--h','2048','--l','8192','--count','128','--offset',str(offset)]
 proc=subprocess.run(cmd)
 receipt=json.loads((E/f'h2048_b{offset:04d}_processes.json').read_text());receipts.append(receipt)
 (E/'scale_capture_processes.json').write_text(json.dumps(receipts,indent=2))
 assert proc.returncode==0 and not receipt['timed_out'],offset
 # Gate rechecks CUDA; allow only this completed block's contexts to disappear naturally.
 for attempt in range(30):
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if not active:break
  time.sleep(2)
 else:raise RuntimeError('compute processes remain after completed capture block')
for replica in range(8):
 out=E/f'infer_h2048_replica{replica}';out.mkdir()
 for name in ['short.json','requests.json','repeat_0_continuations.json','repeat_1_continuations.json']:
  combined=[]
  for offset in [0,128,256,384]:combined.extend(json.loads((E/f'infer_h2048_b{offset:04d}_replica{replica}'/name).read_text()))
  (out/name).write_text(json.dumps(combined))
(E/'h2048_processes.json').write_text(json.dumps(dict(commands=[r['commands'] for r in receipts],uuids=receipts[0]['uuids'],codes=[r['codes'] for r in receipts],seconds=sum(r['seconds'] for r in receipts),peak_sampled_mib=[max(r['peak_sampled_mib'][i] for r in receipts) for i in range(8)],timed_out=False),indent=2))
