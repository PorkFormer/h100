"""Check completed service barriers, provenance, and eight-rank trainer receipts."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence';result=[]
expected=json.loads((E/'h2048_processes.json').read_text())['uuids']
for gate in E.glob('cuda_gate_*.json'):
 g=json.loads(gate.read_text())
 assert [d['uuid'] for d in g['devices']]==expected,gate
for p in sorted(E.glob('train_*_v6/process.json')):
 case=p.parent;proc=json.loads(p.read_text())
 if proc.get('timed_out'):
  result.append(dict(case=case.name,status='FAIL_TIMEOUT',scope='No overall pass; inspect partial receipts and owned cleanup separately'))
  continue
 audit=case/'worker_audit';records=[(x,json.loads(x.read_text())) for x in audit.glob('*.json')]
 imports=[v for x,v in records if x.name.endswith('_imports.json')]
 assert len(imports)>=8,case
 for v in imports:
  for key in ['engine_source','worker_source']:assert Path(v[key]).is_relative_to(R.parent/'verl'),v
 modules=[v for x,v in records if x.name.endswith('_module.json')]
 for v in modules:
  if v['module'].startswith('verl.'):
   assert Path(v['source']).is_relative_to(R.parent/'verl'),v
 devices=[v for x,v in records if x.name.endswith('_actual_device.json')]
 assert len({v['actual_uuid'] for v in devices})==8,case
 starts=[v['version'] for x,v in records if x.name.endswith('_publish_start.json')]
 injected=[v for x,v in records if v.get('event')=='fault_injected']
 steps=[v for x,v in records if v.get('event')=='actual_AdamW_step']
 if case.name.startswith('train_standard_') and not any(t in case.name for t in ['_fault_','_live_oracle_','_ref_']):
  for pid in {v['pid'] for v in steps}:
   per=[v for x,v in sorted(records) if v.get('event')=='actual_AdamW_step' and v['pid']==pid]
   assert len(per)==4,case
 if '_fault_' in case.name:
  assert injected and proc['code']!=0 and not steps and all(v==0 for v in starts),case
  if '_fault_release_' in case.name:
   assert not any(v.get('event')=='sleep_attempt' and v['time_ns']>=min(i['time_ns'] for i in injected) for x,v in records),case
 if '_live_oracle_' in case.name:
  assert proc['code']!=0 and not steps and all(v==0 for v in starts),case
  complete=[v for x,v in records if x.name.endswith('_live_validation_complete.json')]
  assert len(complete)==1,case
  original=[v for x,v in records if x.name.endswith('_live_oracle.json')]
  assert len(original)==1 and original[0]['release_intervals']==len(original[0]['requests']),case
 result.append(dict(case=case.name,status='PASS',engine_worker_import_receipts=len(imports),module_receipts=len(modules),actual_gpu_count=8,publish_start_versions=sorted(set(starts)),remote_release_ack_count=sum(v.get('event')=='remote_release_ack' for x,v in records),remote_drain_ack_count=sum(v.get('event')=='remote_drain_ack' for x,v in records)))
(E/'receipt_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
