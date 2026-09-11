"""Summarize actual updates and prefix identity evidence; missing evidence stays uncovered."""
import collections,json,sys,math
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'verl'))
summary=[]
for case in sorted((ROOT/'evidence').glob('train_*_v6')):
    process=case/'process.json'
    if not process.exists():continue
    proc=json.loads(process.read_text());audit=case/'worker_audit';steps=[];pub=[];devices=[]
    for p in audit.glob('*.json'):
        value=json.loads(p.read_text())
        if value.get('event')=='actual_AdamW_step':steps.append(value)
        if p.name.endswith('_publish_done.json'):pub.append(value['version'])
        if p.name.endswith('_actual_device.json'):devices.append(value)
    counts=collections.Counter(x['pid'] for x in steps)
    known={};versions=[];candidate_count=0;raw_counter=collections.Counter()
    def token_key(b,i):
        pw=b.batch['prompts'].shape[-1];pm=b.batch['attention_mask'][i,:pw].bool();rm=b.batch['response_mask'][i].bool()
        return (tuple(b.batch['prompts'][i][pm].tolist()),tuple(b.batch['responses'][i][rm].tolist()))
    for p in audit.glob('*rollout_output.pt'):
        b=torch.load(p,weights_only=False)
        for i in range(len(b)):raw_counter[token_key(b,i)]+=1
        versions.extend(np.asarray(b.non_tensor_batch.get('global_steps',[])).reshape(-1).tolist())
    for p in audit.glob('*hook_input.pt'):
        b=torch.load(p,weights_only=False)['batch']
        for i in range(len(b)):
            key=(str(b.non_tensor_batch['uid'][i]),str(b.non_tensor_batch['trajectory_id'][i]));known[key]=b.batch['responses'][i][b.batch['response_mask'][i].bool()]
        versions.extend(np.asarray(b.non_tensor_batch.get('rollout_policy_version',[])).reshape(-1).tolist())
    for p in sorted(audit.glob('*candidate_before.pt')):
        b=torch.load(p,weights_only=False);candidate_count+=1
        for i in range(len(b)):
            key=(str(b.non_tensor_batch['uid'][i]),str(b.non_tensor_batch['trajectory_id'][i]))
            mask=b.batch['attention_mask'][i,-b.batch['responses'].shape[-1]:].bool();known[key]=b.batch['responses'][i][mask]
        versions.extend(np.asarray(b.non_tensor_batch.get('rollout_policy_version',[])).tolist())
    actor_batches=0;verified_rows=0;actor_missing=0;multiset_rows=0
    for p in audit.glob('*trainer_actor_batch.pt'):
        b=torch.load(p,weights_only=False);actor_batches+=1;assert b.batch['responses'].shape[-1]<=2048
        for i in range(len(b)):
            key=(str(b.non_tensor_batch['uid'][i]),str(b.non_tensor_batch.get('trajectory_id',['']*len(b))[i]));mask=b.batch['response_mask'][i].bool()
            if key in known:
                assert torch.equal(known[key],b.batch['responses'][i][mask]);verified_rows+=1
            elif raw_counter[token_key(b,i)]>0:multiset_rows+=1
            else:actor_missing+=1
            if raw_counter[token_key(b,i)]>0:raw_counter[token_key(b,i)]-=1
    hook_calls=0;continuations=0;release_intervals=0;padded_chunks=[];changed_rows=0
    for p in audit.glob('*hook_output.pt'):
        result=torch.load(p,weights_only=False);hook_calls+=1
        before=torch.load(p.with_name(p.name.replace('_hook_output','_hook_input')),weights_only=False)
        changed_rows+=int(result.corrected_mask.sum()) if result.corrected_mask is not None else 0
        raw=before['raw_scores'];delta=result.effective_scores-raw;mask=before['batch'].batch['response_mask'];allowed=torch.zeros_like(delta,dtype=torch.bool);allowed[torch.arange(len(raw)),mask.sum(-1).long()-1]=True
        assert torch.equal(delta[~allowed],torch.zeros_like(delta[~allowed]))
        if result.aux:
            capture=result.aux['capture'];continuations+=len(capture.requests)
            padded_chunks.extend(x.metadata for x in result.aux['result'].profiling_intervals if x.name=='long_reward_model_forward')
            release_intervals+=sum(x.name=='continuation_cleanup_release' for x in capture.profiling_intervals)
    assert continuations==release_intervals
    loss_metrics=[];memory_metrics=[]
    for p in audit.glob('*trainer_actor_result.pt'):
        metrics=torch.load(p,weights_only=False).meta_info['metrics']
        for key in ['actor/loss','actor/pg_loss','actor/grad_norm']:
            values=np.asarray(metrics[key],dtype=float);assert np.isfinite(values).all(),(case.name,key)
        loss_metrics.append({k:np.asarray(v).tolist() for k,v in metrics.items() if k in ['actor/loss','actor/pg_loss','actor/grad_norm']})
        memory_metrics.append({k:np.asarray(v).tolist() for k,v in metrics.items() if 'max_memory_' in k})
    log=(case/'run.log').read_text();exhausted='Generated too many; data may be too difficult.' in log
    logical_steps=min(counts.values()) if len(counts)==8 else 0
    if counts:assert len(set(counts.values()))==1 and len(counts)==8,counts
    if proc['code']==0 and 'train_standard' in case.name and '_fault_' not in case.name:
        expected=1 if '_ref_' in case.name or '_live_oracle_smoke_' in case.name else 2
        assert logical_steps==expected,(case.name,logical_steps,expected)
    status='UNCOVERED_DAPO_EXHAUSTION' if exhausted else ('PASS' if proc['code']==0 else 'FAIL')
    for suite, key, marker in [('live_oracle_summary.json','horizon','_live_oracle_'),('fault_summary.json','fault','_fault_')]:
        receipt=ROOT/'evidence'/suite
        if marker in case.name and receipt.exists():
            matches=[r for r in json.loads(receipt.read_text()) if marker+str(r[key])+'_v6' in case.name]
            if len(matches)==1:status=matches[0]['status']
    if proc.get('timed_out'):status='FAIL_TIMEOUT'
    summary.append(dict(case=case.name,process_code=proc['code'],status=status,logical_optimizer_steps=logical_steps,optimizer_step_receipts=len(steps),weight_changed_receipts=sum(s['before']!=s['after'] for s in steps),published_versions=sorted(set(pub)),observed_normal_versions=sorted(set(versions)),actual_gpu_uuids=sorted({x['actual_uuid'] for x in devices}),candidate_batches=candidate_count,actor_batches=actor_batches,actor_rows_verified_against_short=verified_rows,actor_rows_verified_by_token_multiset=multiset_rows,actor_rows_without_candidate_receipt=actor_missing,real_corrected_rows=changed_rows,long_reward_chunk_metadata=padded_chunks,hook_calls=hook_calls,continuation_requests=continuations,acknowledged_release_intervals=release_intervals,loss_metrics=loss_metrics,actor_memory_metrics=memory_metrics,seconds=proc['seconds']))
(ROOT/'evidence/case_analysis.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
