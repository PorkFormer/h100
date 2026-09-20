#!/usr/bin/env python3
"""Manual Qwen3-4B evaluation using the frozen NCBR 100-prompt protocol."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
TEMPLATES=HERE/'templates'
TOKENIZER=Path('/workspace/models/Qwen3-4B-Base')
VERIFIER=Path('/workspace/rl/h100-natural-continuation-boundary-return-v1/verl/verl/utils/reward_score/math_dapo.py')

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(32<<20),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def check_package():
    for name,digest in read(HERE/'package_sha256.json').items():
        if sha(HERE/name)!=digest:raise ValueError('package changed: '+name)

    for item in read(HERE/'input_provenance.json')['files']:
        if sha(item['source'])!=item['sha256']:raise ValueError('frozen local evaluation input changed: '+item['source'])
    for path,digest in read(HERE/'input_provenance.json')['external_assets_sha256'].items():
        if sha(path)!=digest:raise ValueError('historical tokenizer/verifier changed: '+path)

def checkpoint_manifest(hf,model_id,step,label):
    from safetensors import safe_open
    cfg=read(hf/'config.json');base=read(TOKENIZER/'config.json')
    for k in ('model_type','architectures','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size','head_dim','eos_token_id','bos_token_id','tie_word_embeddings','rope_theta'):
        if cfg.get(k)!=base.get(k):raise ValueError('checkpoint incompatible with frozen Qwen3-4B protocol: '+k)
    files=sorted(hf.glob('*.safetensors'))
    if not files:raise ValueError('no safetensors weight files')
    index=hf/'model.safetensors.index.json';mapping=read(index)['weight_map'] if index.exists() else None
    if mapping is None and len(files)!=1:raise ValueError('multiple safetensors files require an index')
    if mapping is not None and set(mapping.values())!={p.name for p in files}:raise ValueError('weight shard coverage mismatch')
    keys={}
    for p in files:
        with safe_open(p,framework='pt',device='cpu') as f:
            for k in f.keys():
                if k in keys:raise ValueError('duplicate tensor: '+k)
                keys[k]=p.name;f.get_slice(k).get_shape()
    if not keys:raise ValueError('empty weights')
    if mapping is not None and keys!=mapping:raise ValueError('tensor index mismatch')
    weights=[]
    for p in sorted(hf.iterdir()):
        if p.is_file():
            before=p.stat();digest=sha(p);after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('checkpoint changed during hashing')
            weights.append(dict(path=str(p),size=after.st_size,mtime_ns=after.st_mtime_ns,sha256=digest))
    identity=hashlib.sha256(json.dumps(weights,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(checkpoints=[dict(model_id=model_id,arm=label,step=step,checkpoint_path=str(hf),sha256=identity,weight_files=weights)],selection='explicit manual checkpoint; no automatic training dependency')

def prepare(args):
    check_package()
    hf=args.checkpoint.expanduser().resolve(strict=True);out=args.output.expanduser().resolve()
    if not hf.is_dir():raise ValueError('--checkpoint must be an HF directory')
    if out.exists():raise ValueError('output already exists; choose a new directory (start an existing prepared run with start)')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.model_id):raise ValueError('model-id must contain letters, digits, _, . or -')
    if args.step<0:raise ValueError('step must be nonnegative; supplied explicitly, never inferred from filename')
    print('Checking and hashing checkpoint (CPU only)...',flush=True)
    manifest=checkpoint_manifest(hf,args.model_id,args.step,args.label)
    # No existing result directory can be overwritten, including a concurrent prepare.
    out.mkdir(parents=True,exist_ok=False)
    run_id=hashlib.sha256(str(out).encode()).hexdigest()[:12]
    config=dict(node=args.node,ray_address=args.ray_address,namespace='ncbr-manual-'+run_id,cache_root='/tmp/nce/'+run_id,port_base=20000+(int(run_id[:4],16)%24000),model_id=args.model_id,step=args.step)
    for name in read(HERE/'package_sha256.json'):
        if name.startswith('templates/'):
            p=HERE/name;shutil.copy2(p,out/p.name)
    for item in read(HERE/'input_provenance.json')['files']:
        if item['name'] not in ('prompts.parquet','prompt_manifest.json','seeds.json'):
            raise ValueError('unexpected protocol input')
        shutil.copy2(item['source'],out/item['name'])
    write(out/'run_config.json',config);write(out/'checkpoint_manifest.json',manifest)
    protocol=read(out/'generation_config.json');protocol.update(checkpoint_steps=[args.step],target_experiment=args.model_id)
    for h in (2048,8192):protocol['horizons'][f'h{h}']['output_root']=str(out/f'formal/generation/h{h}')
    write(out/'generation_config.json',protocol)
    assets=[VERIFIER]+[p for p in TOKENIZER.iterdir() if p.is_file() and p.suffix in ('.json','.jinja')]
    write(out/'input_provenance.json',dict(package=str(HERE),package_sha256=sha(HERE/'package_sha256.json'),frozen_protocol=read(HERE/'input_provenance.json'),prepared_at=time.time(),python=sys.executable))
    files=sorted(p for p in out.iterdir() if p.is_file())+assets
    write(out/'freeze.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in files]))
    write(out/'status.json',dict(stage='PREPARED',time=time.time(),generation_started=False))
    print(json.dumps(dict(output=str(out),stage='PREPARED',checkpoint=str(hf),rows=6400,node=args.node),ensure_ascii=False))
    return out

def verify_prepared(out):
    for item in read(out/'freeze.json')['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('frozen file changed: '+item['path'])
    manifest=read(out/'checkpoint_manifest.json')
    if len(manifest['checkpoints'])!=1:raise ValueError('exactly one checkpoint is required')
    for w in manifest['checkpoints'][0]['weight_files']:
        st=Path(w['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(w['size'],w['mtime_ns']):raise ValueError('checkpoint metadata changed: '+w['path'])
    if read(out/'status.json').get('stage')!='PREPARED':raise ValueError('run already started or failed; automatic controller restart is not supported')

def start(out,foreground=False):
    out=out.expanduser().resolve(strict=True)
    with (out/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any((out/n).exists() for n in ('launch_receipt.json','driver.json','dispatch.json')):raise ValueError('already launched; refusing duplicate run')
        verify_prepared(out)
        cfg=read(out/'run_config.json');cache=Path(cfg['cache_root'])/'controller';cache.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'PYTHONPYCACHEPREFIX':str(cache/'py'),'XDG_CACHE_HOME':str(cache/'xdg'),'PYTHONPATH':str(out),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        with (out/'driver.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,str(out/'driver.py')],cwd=out,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'launch_receipt.json',dict(pid=proc.pid,time=time.time(),command=[sys.executable,str(out/'driver.py')],foreground_wait=foreground))
    print(json.dumps(dict(driver_pid=proc.pid,output=str(out),status_file=str(out/'status.json')),ensure_ascii=False))
    if foreground:return proc.wait()
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('prepare','run'):
        p=sub.add_parser(action,help='CPU prepare only' if action=='prepare' else 'prepare and launch in background')
        p.add_argument('--checkpoint',type=Path,required=True,help='completed Hugging Face safetensors directory')
        p.add_argument('--output',type=Path,required=True,help='new independent output directory')
        p.add_argument('--model-id',required=True);p.add_argument('--step',type=int,required=True)
        p.add_argument('--label',default='NCBR');p.add_argument('--node',default='10.8.191.127')
        p.add_argument('--ray-address',default='10.8.191.127:6397',help='existing Ray address; never starts a new cluster')
        if action=='run':p.add_argument('--foreground',action='store_true',help='wait for detached driver and return its exit code')
    p=sub.add_parser('start',help='launch an existing PREPARED directory once');p.add_argument('--output',type=Path,required=True);p.add_argument('--foreground',action='store_true')
    p=sub.add_parser('status',help='read status and receipts; no GPU/Ray access');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action in ('prepare','run'):
        out=prepare(args)
        return start(out,args.foreground) if args.action=='run' else 0
    if args.action=='start':return start(args.output,args.foreground)
    out=args.output.expanduser().resolve(strict=True)
    print(json.dumps({n:read(out/n) for n in ('status.json','driver.json','driver_exit.json','resource_release.json') if (out/n).exists()},ensure_ascii=False,indent=2));return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError) as exc:print('ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
