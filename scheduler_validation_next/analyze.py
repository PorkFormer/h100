"""Fixed-work eligibility, weighted occupancy, and paired performance statistics."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;E=ROOT/'evidence'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def weighted_percentile(values,weights,q):
    ix=np.argsort(values);v=np.array(values)[ix];w=np.array(weights)[ix];return float(v[np.searchsorted(np.cumsum(w),sum(w)*q,side='left')])
def analyze(case):
    import re
    log=(E/f'{case}.log').read_text()
    assert not re.search(r'CUDA out of memory|OutOfMemoryError|WorkerCrashedError|ActorDiedError|EngineDeadError',log),'Backend failure in raw log'
    p=E/case;r=json.loads((p/'result.json').read_text());cap=json.loads((p/'capture.json').read_text());events=json.loads((p/'events.json').read_text());process=json.loads((E/f'{case}_process.json').read_text())
    servers=json.loads((p/'servers.json').read_text())
    assert len(servers)==8 and len({s['node_id'] for s in servers})==1
    assert {s['cuda_visible'] for s in servers}==set(process['uuids'])
    assert all(s['tp']==1 and s['modules']['verl']==str(ROOT.parent/'verl/verl/__init__.py') for s in servers)
    req=cap['requests'];gen=cap['generations'];effective=json.loads((p/'effective_requests.json').read_text());expected=[x['request_id'] for x in req]
    assert len(set(expected))==len(expected)==r['request_count']
    assert [x['request_id'] for x in effective]==expected
    records={};active=set();generating=set();dispatched=[];prev=r['monotonic_start'];slot_area=live_area=idle=pending_idle=pending_wall=queue_area=pending_generation_idle=pending_generation_wall=0.;values=[];slots=[];weights=[]
    def interval(now):
        nonlocal prev,slot_area,live_area,idle,pending_idle,pending_wall,queue_area,pending_generation_idle,pending_generation_wall
        dt=now-prev;assert dt>=0
        slot_area+=dt*len(active);live_area+=dt*len(generating);idle+=dt*(r['concurrency']-len(active))
        pending=r['request_count']-len(dispatched);queue_area+=dt*pending
        if pending:
            pending_generation_idle+=dt*(r['concurrency']-len(generating))
            if len(generating)<r['concurrency']:pending_generation_wall+=dt
            pending_idle+=dt*(r['concurrency']-len(active))
            if len(active)<r['concurrency']:pending_wall+=dt
        values.append(len(generating));slots.append(len(active));weights.append(dt);prev=now
    for e in events:
        interval(e['time']);k=e['event'];rid=e['request_id']
        if k=='dispatch':
            assert rid not in records;records[rid]={};active.add(rid);generating.add(rid);dispatched.append(rid)
        assert rid in active and k not in records[rid]
        records[rid][k]=e['time']
        if k=='acquire':records[rid]['server_id']=e['server_id']
        if k=='generation_complete':generating.remove(rid)
        if k=='release_ack':active.remove(rid)
        assert len(active)<=r['concurrency']
    assert dispatched==expected and not active and not generating
    interval(r['monotonic_start']+r['continuation_seconds'])
    per=[]
    for rid,v in records.items():
        assert v['dispatch']<=v['acquire']<=v['generation_complete']<=v['release_start']<=v['release_ack']
        per.append(dict(request_id=rid,server_id=v['server_id'],admission_queue=v['dispatch']-r['monotonic_start'],dispatch=v['acquire']-v['dispatch'],HOL=v['release_start']-v['generation_complete'],release=v['release_ack']-v['release_start'],latency=v['generation_complete']-v['dispatch']))
    dispatch_gaps=np.diff([records[rid]['dispatch'] for rid in expected]).tolist()
    observed_engine_request_seconds={str(i):sum(v['generation_complete']-v['acquire'] for v in records.values() if str(v['server_id'])==str(i)) for i in range(8)}
    assert json.loads((p/'cleanup.json').read_text())['total_inflight']==0
    owned=json.loads((E/f'{case}_owned_cleanup.json').read_text());assert not owned['remaining'] and not owned['errors'];assert process['code']==0 and not process['timed_out']
    lens=[len(g['tail_token_ids']) for g in gen];by={g['request_id']:len(g['tail_token_ids']) for g in gen};assert set(by)==set(expected)
    if not r['natural']:
        assert all(by[x['request_id']]==x['sampling_params']['max_tokens'] and x['sampling_params']['ignore_eos'] for x in effective)
    total=sum(lens);prefill=sum(len(x['input_token_ids']) for x in req);assert total==r['tokens'] and prefill==r['prefill_tokens']
    samples=[s for s in process['samples'] if r['epoch_start']<=s['epoch']<=r['epoch_end']];assert samples
    util=np.array([s['utilization'] for s in samples]);mem=np.array([s['mib'] for s in samples]);T=r['continuation_seconds']
    timing={}
    for name in ['continuation_queue','continuation_prefill_engine','continuation_decode_engine','continuation_cleanup_release']:
        xs=[i['wall_end']-i['wall_start'] for i in cap['intervals'] if i['name']==name]
        timing[name]=dict(available=bool(xs),count=len(xs),request_seconds=sum(xs) if xs else None)
    workload=dict(request_ids=expected,inputs=[x['input_token_ids'] for x in req],decode_lengths=[by[x] for x in expected],prefill_tokens=prefill,decoded_tokens=total,config_sha256=hashlib.sha256((p/'service_config.yaml').read_bytes()).hexdigest(),concurrency=r['concurrency'],batch_size=r['batch_size'],uuid_order=process['uuids'],sampling=[x['sampling_params'] for x in effective],topology=sorted((s['cuda_visible'],s['tp']) for s in servers),modules=servers[0]['modules'])
    summary=dict(**r,case=case,audit='PASS',eligibility_signature=digest(workload),request_per_second=len(req)/T,decoded_tokens_per_second=total/T,total_tokens_per_second=(prefill+total)/T,seconds_per_request=T/len(req),seconds_per_1k_decode=T/total*1000,seconds_per_1k_total=T/(total+prefill)*1000,active_mean=live_area/T,active_p50=weighted_percentile(values,weights,.5),active_p90=weighted_percentile(values,weights,.9),slot_mean=slot_area/T,generation_slot_utilization=live_area/(T*r['concurrency']),generation_idle_slot_seconds=T*r['concurrency']-live_area,pending_generation_idle_slot_seconds=pending_generation_idle,pending_generation_idle_wall_seconds=pending_generation_wall,slot_utilization=slot_area/(T*r['concurrency']),idle_slot_seconds=idle,pending_idle_slot_seconds=pending_idle,pending_but_idle_wall_seconds=pending_wall,admission_queue_request_seconds=queue_area,HOL_request_seconds=sum(x['HOL'] for x in per),release_request_seconds=sum(x['release'] for x in per),dispatch_request_seconds=sum(x['dispatch'] for x in per),request_latency_p90=float(np.percentile([x['latency'] for x in per],90)),gpu_util_mean=float(util.mean()),gpu_util_per_device_mean=util.mean(axis=0).tolist(),gpu_util_per_device_p50=np.percentile(util,50,axis=0).tolist(),gpu_util_per_device_p90=np.percentile(util,90,axis=0).tolist(),peak_sampled_mib=mem.max(axis=0).tolist(),length_quantiles=np.percentile(lens,[0,25,50,75,90,95,100]).tolist(),engine_timing=timing,tail_drain_wall_seconds=r['monotonic_start']+T-records[expected[-1]]['dispatch'],dispatch_gap_p50=float(np.percentile(dispatch_gaps,50)),dispatch_gap_p90=float(np.percentile(dispatch_gaps,90)),dispatch_gap_max=max(dispatch_gaps),observed_per_service_active_mean={k:v/T for k,v in observed_engine_request_seconds.items()},per_request=per,active_engine_mapping=json.loads((p/'servers.json').read_text()))
    (p/'analysis.json').write_text(json.dumps(summary,indent=2));return summary

def paired(pairs):
    ratios=[]
    for a,b in pairs:
        assert a['eligibility_signature']==b['eligibility_signature'],'workload mismatch'
        assert a['audit']==b['audit']=='PASS'
        ratios.append(a['continuation_seconds']/b['continuation_seconds'])
    rng=np.random.default_rng(42);boot=np.array(ratios)[rng.integers(0,len(ratios),size=(10000,len(ratios)))].mean(axis=1);lo,hi=np.percentile(boot,[2.5,97.5]);mean=float(np.mean(ratios))
    return dict(ratios=ratios,mean=mean,median=float(np.median(ratios)),ci95=[float(lo),float(hi)],eligibility='PASS',status='PASS' if len(pairs)>=4 and mean>=1.05 and lo>1 else 'FAIL',pairs=len(pairs))
if __name__=='__main__':
    import sys
    print(json.dumps({k:v for k,v in analyze(sys.argv[1]).items() if k!='per_request'},indent=2))
