"""Read-only final binding of protected source and completed evidence."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
pre=json.loads((R/'preflight.json').read_text())
checks={}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
assert sha(R/'workload.json')==json.loads((R/'workload_manifest.json').read_text())['sha256']
for name,expected in pre['model_sha256'].items():assert sha(Path(pre['model'])/name)==expected,name
for name,v in pre['validated_source'].items():
    now=hashlib.sha256((R.parent/name).read_bytes()).hexdigest();checks[name]=now;assert now==v['actual']
assert not subprocess.check_output(['git','diff','--name-only'],cwd=R.parent,text=True).strip(),'Tracked source changed'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R.parent,text=True).strip()==pre['sha']
for p in (R/'evidence').glob('*_source_hashes.json'):
    for name,expected in json.loads(p.read_text()).items():
        if '/natural_continuation_boundary_return/' in name:assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
with (R/'evidence/final_cuda_gate.log').open('x') as f:subprocess.run(['python',str(R/'gate.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'evidence/final_nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
files={}
for p in sorted(R.rglob('*')):
    if p.is_file() and 'cache' not in p.relative_to(R).parts and '__pycache__' not in p.relative_to(R).parts and p.name!='final_manifest.json':files[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'final_manifest.json').write_text(json.dumps(dict(status='PASS',protected_source=checks,files=files),indent=2));print('FINAL_BINDING_PASS',len(files))
