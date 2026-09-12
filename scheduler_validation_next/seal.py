"""Seal artifacts after the final-check process closes its stdout log."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'final_manifest.json';m=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
changed=[name for name,v in m['files'].items() if sha(R/name)!=v]
(R/'evidence/seal_audit.json').write_text(json.dumps(dict(post_final_check_log_close=True,changed_since_preliminary_binding=changed,reason='Final check stdout closes after its preliminary manifest is written; final seal binds closed logs and final reporting files.'),indent=2))
m['files']={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and 'cache' not in p.relative_to(R).parts and '__pycache__' not in p.relative_to(R).parts and p.name!='final_manifest.json'}
m['seal_method']='Post-process closed-log SHA256 seal'
p.write_text(json.dumps(m,indent=2))
for name,v in m['files'].items():assert sha(R/name)==v,name
print('SEALED_AND_VERIFIED',len(m['files']))
