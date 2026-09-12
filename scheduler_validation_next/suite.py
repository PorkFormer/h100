"""Sequential fresh-service AB/BA experiment. Never starts training."""
import json,subprocess,sys
from pathlib import Path
from analyze import analyze,paired
ROOT=Path(__file__).resolve().parent;E=ROOT/'evidence'
def run(name,c,scheduler='work_conserving',count=256,natural=False):
    receipt=E/f'{name}_process.json'
    if not receipt.exists():
        # An incomplete previous case is evidence, never silently overwritten or resumed.
        assert not (E/f'{name}.log').exists(),f'Incomplete instance: {name}'
        cmd=[sys.executable,str(ROOT/'launch.py'),'--case',name,'--concurrency',str(c),'--scheduler',scheduler,'--count',str(count)]+(['--natural'] if natural else [])
        with (E/f'{name}_launch.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    r=analyze(name);print(json.dumps(dict(case=name,seconds=r['continuation_seconds'],tok_s=r['decoded_tokens_per_second'])),flush=True);return r
sweep=[run(f'sweep_c{c}',c,count=128) for c in [8,16,32,64,128]]
if '--sweep-only' in sys.argv:
    (ROOT/'sweep.json').write_text(json.dumps(sweep,indent=2))
    sys.exit(0)
best=max(r['decoded_tokens_per_second'] for r in sweep)
csat=min(r['concurrency'] for r in sweep if r['decoded_tokens_per_second']>=.95*best)
comparisons=sorted(set([csat,128]))
if len(comparisons)==1:comparisons.insert(0,max(r['concurrency'] for r in sweep if r['concurrency']<csat))
selection=dict(C_sat=csat,high_load=128,comparison_concurrencies=comparisons,additional_bracket=csat==128,rule='Lowest successful concurrency at >=95% maximum observed decode throughput; all selected instances audited; latency and HBM are reported for manual stability assessment',sweep=[{k:v for k,v in r.items() if k!='per_request'} for r in sweep])
(ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
for c in comparisons:
    pairs=[]
    for i in range(4):
        order=['fixed_wave','work_conserving'] if i%2==0 else ['work_conserving','fixed_wave'];results={}
        for s in order:results[s]=run(f'pair_c{c}_{i}_{s}',c,s)
        pairs.append((results['fixed_wave'],results['work_conserving']))
        (ROOT/f'paired_c{c}.json').write_text(json.dumps(paired(pairs),indent=2))
for c in comparisons:
    for s in ['fixed_wave','work_conserving']:run(f'natural_c{c}_{s}',c,s,count=128,natural=True)
print('EXPERIMENTS_COMPLETE',flush=True)
