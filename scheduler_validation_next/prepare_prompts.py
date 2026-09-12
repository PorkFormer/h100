import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from verl.utils.tokenizer import normalize_token_ids
old=Path('/tmp/ncbr_8gpu_scale_20260911/validation/evidence/prompts.jsonl')
rows=[json.loads(x) for x in old.read_text().splitlines()];seen={r['uid'] for r in rows}
tok=AutoTokenizer.from_pretrained('/workspace/models/Qwen3-1.7B-Base',local_files_only=True)
# Inspect the full unique block; exclude conflicting labels before deterministic selection.
raw=pq.ParquetFile('/workspace/rl/data/dapo_math_17k_train.parquet').read_row_group(0).to_pylist()[:17917]
unique={};conflicts=set()
for r in raw:
    key=json.dumps(r['prompt'],ensure_ascii=False,sort_keys=True);uid=hashlib.sha256(key.encode()).hexdigest()
    if uid in unique and unique[uid]['reward_model']['ground_truth']!=r['reward_model']['ground_truth']:conflicts.add(uid)
    unique.setdefault(uid,r)
for uid,r in unique.items():
    if uid in seen or uid in conflicts:continue
    ids=normalize_token_ids(tok.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True))
    if not ids or len(ids)>1024:continue
    rows.append(dict(uid=uid,prompt_token_ids=ids));seen.add(uid)
    if len(rows)==3072:break
assert len(rows)==3072
with (ROOT/'prompts.jsonl').open('x') as f:
    for r in rows:f.write(json.dumps(r)+'\n')
print('PROMPT_POOL',len(rows))
