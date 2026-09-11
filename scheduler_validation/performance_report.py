"""Paired performance gate, independent of frozen functional correctness."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;E=R/'evidence'
def load(case):
    r=json.loads((E/case/'result.json').read_text())
    c=json.loads((E/case/'capture.json').read_text())
    events=json.loads((E/case/'events.json').read_text())
    tokens=[(g['request_id'],g['branch_id'],g['tail_token_ids'],g['stop_reason'],g['finish_reason']) for g in c['generations']]
    active=0;peak=0;idle=0.;last=None;dispatched=0
    trace=[];releases={};release_times=[]
    for event in events:
        t=event['time']
        if last is not None and dispatched<78 and active<r['concurrency']:idle+=t-last
        if event['event']=='dispatch':active+=1;dispatched+=1
        if event['event']=='release_start':releases[event['request_id']]=t
        if event['event']=='release_ack':
            active-=1;release_times.append(t-releases[event['request_id']])
        peak=max(peak,active);last=t;trace.append([t,active])
    assert active==0 and peak<=r['concurrency']
    completed={i['metadata']['request_id']:i['wall_end'] for i in c['intervals'] if i['name']=='continuation_request'}
    completed_wait={rid:max(0.,start-completed[rid]) for rid,start in releases.items()}
    process=json.loads((E/f'{case}_process.json').read_text())
    r.update(completed_waiting_release_seconds=completed_wait,peak_active=peak,refillable_idle_seconds=idle,release_seconds=release_times,
             active_trace=trace,peak_sampled_mib=process['peak_sampled_mib'])
    (E/case/'metrics.json').write_text(json.dumps(r,indent=2))
    return r,c['requests'],tokens

def report():
    output={}
    try:
        _,rq0,t0=load('repeat_old_0');_,rq1,t1=load('repeat_old_1')
        output['old_repeat']=dict(exact=rq0==rq1 and t0==t1,
            different_requests=sum(a!=b for a,b in zip(t0,t1,strict=True)))
    except FileNotFoundError:output['old_repeat']=dict(status='UNCOVERED')
    for group in ['scheduler','capacity']:
        pairs=[];missing=[]
        for i in range(6):
            names=[f'{group}_{i}_{arm}' for arm in ['a','b']]
            try:
                a,ra,ta=load(names[0]);b,rb,tb=load(names[1])
                assert ra==rb
                pairs.append(dict(index=i,seconds_a=a['continuation_seconds'],seconds_b=b['continuation_seconds'],
                    ratio=a['continuation_seconds']/b['continuation_seconds'],tokens_a=a['tokens'],tokens_b=b['tokens'],
                    exact_work=ta==tb,different_requests=sum(x!=y for x,y in zip(ta,tb,strict=True))))
            except FileNotFoundError:missing.append(i)
        ratios=np.asarray([p['ratio'] for p in pairs]);ci=None;mean=None
        if len(pairs)==6:
            mean=float(ratios.mean())
            rng=np.random.default_rng(42)
            boot=ratios[rng.integers(0,6,size=(10000,6))].mean(axis=1)
            ci=np.quantile(boot,[.025,.975]).tolist()
        comparable=bool(len(pairs)==6 and all(p['exact_work'] for p in pairs) and output['old_repeat'].get('exact',False))
        passed=bool(comparable and mean>=1.05 and ci[0]>1)
        output[group]=dict(pairs=pairs,missing=missing,work_comparable=comparable,
            diagnostic_ratio_mean=mean,diagnostic_paired_95_ci=ci,performance_pass=passed,
            interpretation='PERFORMANCE_PASS' if passed else 'DIAGNOSTIC_ONLY' if pairs else 'UNCOVERED')
    (E/'performance_report.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
if __name__=='__main__':report()
