"""Bind source, dependencies, original frozen inputs and model before GPU use."""
import hashlib, importlib.metadata, json, subprocess
from pathlib import Path
R = Path(__file__).resolve().parent
old = json.loads((R.parent/'validation/run_manifest.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
for name, expected in old['model_sha256'].items():
    assert sha(Path(old['model'])/name)==expected, name
for name, expected in old['dependencies'].items():
    assert importlib.metadata.version(name)==expected, name
source=Path(old['evidence_root'])
inputs={}
for pattern in ['infer_h2048_replica*/requests.json','infer_h2048_replica*/short.json',
                'infer_h2048_replica*/repeat_0_continuations.json','prompts.parquet','prompts.jsonl',
                'replay_trainer_natural/actor_actual_input.pt','replay_h2048_v2/actor_input.pt']:
    for path in source.glob(pattern):inputs[str(path)]=sha(path)
assert len(list(source.glob('infer_h2048_replica*/requests.json')))==8
requests=[r for p in sorted(source.glob('infer_h2048_replica*/requests.json')) for r in json.loads(p.read_text())]
# Canonical detector order is recovered from the full frozen batch by the runner.
assert len(requests)==78
receipt=dict(base='c557509845f5aacbde9b990ba9157b66502a294b',model=old['model'],
             model_sha256=old['model_sha256'],dependencies=old['dependencies'],inputs=inputs,
             uuid_order=old['uuid_order'],request_count=78,short_rows=2048,H=2048,L=8192,
             concurrency=[4,8],batch_size=8,request_timeout=600,case_timeout=1800,
             bootstrap_seed=42,paired_repetitions=6,default='fixed_wave',
             performance_threshold=dict(speedup=1.05,ci_lower_greater_than=1),
             production_determinism_changes=False)
(R/'manifest.json').write_text(json.dumps(receipt,indent=2))
print('FREEZE PASS')
