"""Bind final raw evidence; never mutate previous evidence or shared runtime state."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
state={}
for name,command in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader']),
                     ('compute_processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),
                     ('recovery',['nvidia-smi','--query-gpu=index,gpu_recovery_action','--format=csv,noheader'])]:
    p=subprocess.run(command,capture_output=True,text=True)
    state[name]=dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
(E/'final_state.json').write_text(json.dumps(state,indent=2))
artifacts={}
for path in sorted(E.rglob('*')):
    if path.is_file() and path.name!='final_manifest.json':
        artifacts[str(path.relative_to(R))]=dict(bytes=path.stat().st_size,sha256=sha(path))
source={str(p.relative_to(R.parent)):sha(p) for p in R.glob('*.py')}
record=dict(time=time.time(),revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
            implementation=json.loads((R/'source_manifest.json').read_text()),audit_fix=json.loads((R/'audit_fix_manifest.json').read_text()),runner_files=source,artifacts=artifacts,
            runtime_directories=[json.loads(p.read_text()).get('runtime_directory', str(Path('/tmp')/('ncs_'+p.name.removesuffix('_process.json')))) for p in E.glob('*_process.json')])
with (E/'final_manifest.json').open('x') as f:f.write(json.dumps(record,indent=2))
print(f'Bound {len(artifacts)} raw evidence files')
