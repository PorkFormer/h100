"""Read-only input/source binding and complete-instance receipt validation."""
import hashlib,importlib.metadata,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence';manifest=json.loads((R/'manifest.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
for path,digest in manifest['inputs'].items():assert sha(path)==digest,path
for path,digest in manifest['model_sha256'].items():assert sha(Path(manifest['model'])/path)==digest,path
for package,version in manifest['dependencies'].items():assert importlib.metadata.version(package)==version,package
assert json.loads((E/'suite_complete.json').read_text())['status']=='EXECUTED'
expected=['nccl_initial']+[f'actor_{s}_{c}' for s,c in [('fixed_wave',4),('work_conserving',4),('work_conserving',8)]]
expected += [f'repeat_old_{i}' for i in range(2)]
expected += [f'{group}_{i}_{arm}' for group in ['scheduler','capacity'] for i in range(6) for arm in ['a','b']]
expected += [f'train_{entry}_{loss}_{mode}_c{c}' for c in [4,8] for entry,loss,mode in [('dapo','vanilla','shadow'),('dapo','vanilla','replace'),('standard','gspo','replace')]]
for name in expected:
    p=json.loads((E/f'{name}_process.json').read_text());assert p['code']==0 and not p['timed_out'],name
    cleanup=E/f'{name}_owned_cleanup.json'
    if cleanup.exists():
        v=json.loads(cleanup.read_text());assert not v['remaining'] and not v['errors'],name
for r in json.loads((E/'trainer_report.json').read_text()):assert r['status']=='PASS',r
source=json.loads((R/'source_manifest.json').read_text())['files']
assert sha(R.parent/'verl/verl/experimental/natural_continuation_boundary_return/runtime.py')==json.loads((R/'audit_fix_manifest.json').read_text())['runtime_sha256']
for name,digest in source.items():
    if name.endswith(('runtime.py','test_scheduler_on_cpu.py')):continue
    assert sha(R.parent/name)==digest,name
subprocess.run(['git','diff','--quiet','b4077eb','--','verl/tests/experimental/natural_continuation_boundary_return/test_scheduler_on_cpu.py'],cwd=R.parent,check=True)
subprocess.run(['git','diff','--check'],cwd=R.parent,check=True)
result=dict(status='PASS',cases=len(expected),frozen_inputs=len(manifest['inputs']),model_files=len(manifest['model_sha256']),dependencies=manifest['dependencies'],production_sources='exact against implementation plus failure-log audit fix',cases_without_owned_cleanup_receipt=[n for n in expected if not (E/f'{n}_owned_cleanup.json').exists()])
(E/'final_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
