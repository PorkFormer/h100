"""Run dependent replay, actor diagnostic and enlarged trainers after capture completes."""
import argparse,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);a=p.parse_args()
assert (E/'h2048_processes.json').exists(),'Complete scale_capture.py first'
commands=[
 ['replay.py','--h','2048','--l','8192','--suffix','v2'],
 ['launch_actor.py','--h','2048'],
 ['scale_actor_compare.py'],
 ['remaining_cases.py'],
 ['analyze_cases.py'],
 ['scale_coverage.py'],
]
for index,args in enumerate(commands):
 if index<a.start:continue
 with (E/f'scale_stage_{index}_{args[0]}.log').open('x') as f:
  p=subprocess.run([sys.executable,str(R/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,args
