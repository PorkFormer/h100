"""Read-only progress from persisted receipts; no process control."""
import collections,json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
folders=sorted([p for p in E.iterdir() if p.is_dir() and p.name.startswith(('pair_','natural','sweep_'))],key=lambda p:p.stat().st_mtime)
for p in folders[-2:]:
    r=p/'result.json'
    if r.exists():
        x=json.loads(r.read_text());print(p.name,'complete',round(x['continuation_seconds'],3),'seconds')
    else:
        q=p/'events.jsonl'
        print(p.name,dict(collections.Counter(json.loads(x)['event'] for x in q.read_text().splitlines())) if q.exists() else 'startup/warmup')
