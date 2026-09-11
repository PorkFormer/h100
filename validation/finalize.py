"""Record final read-only state after all owned services have exited."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
R=Path(__file__).resolve().parent;E=R/'evidence'
owned=[]
for p in psutil.process_iter(['pid','cmdline','status']):
 try:
  if p.environ().get('NCBR_AUDIT_DIR','').startswith(str(E)+'/') and p.status()!=psutil.STATUS_ZOMBIE:owned.append(p.info)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
assert not owned,owned
state={'time_ns':time.time_ns(),'owned_active_processes':owned}
for key,cmd in [('gpu',['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv']),('compute_processes',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv'])]:state[key]=subprocess.check_output(cmd,text=True)
(E/'final_state.json').write_text(json.dumps(state,indent=2))
base=json.loads((E/'manifest.json').read_text())
base.update(validation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R.parent,text=True).strip(),uuid_order=json.loads((E/'h2048_processes.json').read_text())['uuids'],isolation_root=str(R.parent),production_diff=subprocess.check_output(['git','diff','37a812d','--','verl'],cwd=R.parent,text=True))
assert not base['production_diff']
base['source_sha256']={}
for name in subprocess.check_output(['git','ls-files','verl'],cwd=R.parent,text=True).splitlines():
 p=R.parent/name
 if p.is_file():base['source_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
base['runner_sha256']=json.loads((E/'runner_sha256.json').read_text())
(E/'final_manifest.json').write_text(json.dumps(base,indent=2))
index=[]
for p in sorted(E.rglob('*')):
 if p.is_file() and not any(x in ['cache','__pycache__'] for x in p.relative_to(E).parts) and p.name!='artifact_index.json':index.append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size})
(E/'artifact_index.json').write_text(json.dumps(index,indent=2))
print(json.dumps({'status':'PASS','artifacts_indexed':len(index),'state':state},indent=2))
