import hashlib, importlib.metadata, json, os, subprocess
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence'
OUT.mkdir(exist_ok=True)
for name in ['prompts.parquet','prompts.jsonl','manifest.json']:
    if (OUT/name).exists(): raise FileExistsError(f'Fresh run required; refusing to overwrite {OUT/name}')
model=Path('/workspace/models/Qwen3-1.7B-Base')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
data=Path('/workspace/rl/data/dapo_math_17k_train.parquet')
table=pq.read_table(data)
unique={}; conflicts=set()
for batch in table.to_batches(max_chunksize=20000):
    for row in batch.to_pylist():
        key=json.dumps(row['prompt'],ensure_ascii=False,sort_keys=True)
        gt=json.dumps(row['reward_model']['ground_truth'],sort_keys=True)
        if key in unique and gt!=unique[key][0]: conflicts.add(key)
        elif key not in unique: unique[key]=(gt,row)
eligible=[]
for key,(_,row) in unique.items():
    if key in conflicts: continue
    ids=tok.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True)
    if hasattr(ids,'get'): ids=ids['input_ids']
    if len(ids)<=1024: eligible.append((key,row,ids))
rng=np.random.default_rng(42)
first=rng.choice(len(eligible),128,replace=False)
remaining=np.array([i for i in range(len(eligible)) if i not in set(first)])
indices=np.concatenate([first,rng.choice(remaining,896,replace=False)])
selected=[eligible[i] for i in indices]
pq.write_table(__import__('pyarrow').Table.from_pylist([x[1] for x in selected],schema=table.schema),OUT/'prompts.parquet')
with (OUT/'prompts.jsonl').open('x') as f:
    for i,(key,row,ids) in enumerate(selected):
        f.write(json.dumps(dict(index=i,uid=hashlib.sha256(key.encode()).hexdigest(),prompt=row['prompt'],prompt_token_ids=ids,ground_truth=row['reward_model']['ground_truth']),ensure_ascii=False)+'\n')
manifest=dict(model=str(model),model_sha256={p.name:sha(p) for p in sorted(model.iterdir()) if p.is_file()},data=dict(path=str(data),sha256=sha(data),rows=table.num_rows,unique=len(unique),conflicts=len(conflicts),eligible=len(eligible),seed=42,selection='numpy.default_rng.choice without replacement in first occurrence order',manifest_sha256=sha(OUT/'prompts.jsonl')),commits={k:subprocess.check_output(['git','rev-parse',v],text=True).strip() for k,v in dict(hook='37a812d',oracle='d23da0e',baseline='bfa0886').items()},dependencies={k:importlib.metadata.version(k) for k in ['torch','vllm','ray','transformers','flash-attn','tensordict','numpy','pyarrow']})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
