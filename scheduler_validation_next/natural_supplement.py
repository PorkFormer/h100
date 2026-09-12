"""One bounded queued natural pair: N256>C128, no statistical speedup claim."""
import json,subprocess,sys
from pathlib import Path
from analyze import analyze
R=Path(__file__).resolve().parent;E=R/'evidence'
# The original suite must finish and clean every natural case first.
for c in [64,128]:
    for s in ['fixed_wave','work_conserving']:analyze(f'natural_c{c}_{s}')
for s in ['fixed_wave','work_conserving']:
    name=f'natural256_c128_{s}'
    with (E/f'{name}_launch.log').open('x') as f:
        subprocess.run([sys.executable,str(R/'launch.py'),'--case',name,'--concurrency','128','--scheduler',s,'--count','256','--natural'],stdout=f,stderr=subprocess.STDOUT,check=True)
    r=analyze(name);print(name,r['continuation_seconds'],r['tokens'],flush=True)
print('NATURAL_SUPPLEMENT_COMPLETE',flush=True)
