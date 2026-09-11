"""Audit completed timing traces and summarize slot/request latency without changing runs."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent; E=R/'evidence'

def summarize(folder):
    result=json.loads((folder/'result.json').read_text())
    capture=json.loads((folder/'capture.json').read_text())
    events=json.loads((folder/'events.json').read_text())
    expected=[r['request_id'] for r in capture['requests']]
    active=set();dispatched=[];records={};peak=0;idle=0.;previous=None;cross_batch=False
    for event in events:
        now=event['time'];kind=event['event'];rid=event['request_id']
        if previous is not None and len(dispatched)<len(expected):
            idle+=(now-previous)*(result['concurrency']-len(active))
        previous=now
        if kind=='dispatch':
            assert rid not in records
            index=len(dispatched)
            if index and index%8==0 and active:cross_batch=True
            active.add(rid);dispatched.append(rid);records[rid]={}
            peak=max(peak,len(active));assert peak<=result['concurrency']
        assert rid in active
        assert kind not in records[rid]
        records[rid][kind]=now
        if kind=='acquire':records[rid]['server_id']=event['server_id']
        if kind=='release_ack':active.remove(rid)
    assert dispatched==expected and not active
    rows=[]
    for rid in expected:
        v=records[rid]
        assert v['dispatch']<=v['acquire']<=v['generation_complete']<=v['release_start']<=v['release_ack']
        rows.append(dict(request_id=rid,server_id=v['server_id'],
            dispatch_seconds=v['acquire']-v['dispatch'],
            generation_seconds=v['generation_complete']-v['acquire'],
            completed_waiting_release_seconds=v['release_start']-v['generation_complete'],
            release_seconds=v['release_ack']-v['release_start'],
            slot_seconds=v['release_ack']-v['dispatch']))
    return dict(case=folder.name,scheduler=result['scheduler'],concurrency=result['concurrency'],
        status='PASS',request_order_exact=True,requests=len(rows),peak_slots=peak,
        cross_batch_refill_observed=cross_batch,
        pending_idle_slot_seconds=idle,
        completed_waiting_release_request_seconds=sum(r['completed_waiting_release_seconds'] for r in rows),
        release_request_seconds=sum(r['release_seconds'] for r in rows),per_request=rows)

if __name__=='__main__':
    reports=[summarize(p.parent) for p in sorted(E.glob('*/result.json')) if p.parent.name.startswith(('scheduler_', 'capacity_')) and (p.parent/'events.json').exists() and (p.parent/'capture.json').exists()]
    (E/'event_report.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='per_request'} for r in reports],indent=2))
