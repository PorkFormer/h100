"""Read-only progress from the exclusive run receipts and append-only event log."""
import collections,json,time
from pathlib import Path
E=Path(__file__).resolve().parent/'evidence'
logs=list(E.glob('*_launch.log'))
if logs:
    latest=max(logs,key=lambda p:p.stat().st_mtime)
    name=latest.name.removesuffix('_launch.log')
    result=dict(case=name,seconds_since_launch_log=int(time.time()-latest.stat().st_mtime))
    events=E/name/'events.jsonl'
    if events.exists():
        counts=collections.Counter(json.loads(line)['event'] for line in events.read_text().splitlines() if line)
        result.update(dict(counts))
    receipt=E/f'{name}_process.json'
    if receipt.exists():
        proc=json.loads(receipt.read_text());result.update(code=proc['code'],timed_out=proc['timed_out'],seconds=proc['seconds'])
    if name=='repeat_old_1':
        base={g['request_id']:g for g in json.loads((E/'repeat_old_0/capture.json').read_text())['generations']}
        files=list((E/name).glob('boundary-return-*.json'))
        result['repeat_tokens_compared']=len(files)
        result['repeat_token_differences']=sum(json.loads(p.read_text())[0]['token_ids']!=base[p.stem]['tail_token_ids'] for p in files)
    print(json.dumps(result))
for name in ['performance_report.json','suite_progress.json','suite_complete.json']:
    path=E/name
    if path.exists():
        value=json.loads(path.read_text())
        if name=='performance_report.json':
            value={k:dict(pairs=len(v['pairs']),performance_pass=v['performance_pass']) if 'pairs' in v else v for k,v in value.items()}
        print(name,json.dumps(value))
