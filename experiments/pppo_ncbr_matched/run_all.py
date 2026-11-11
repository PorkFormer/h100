"""No training bypass: execute each gate, then fresh formal S300 and final evaluation."""
import os
import subprocess
import sys
import time
from prepare import HERE, REPO, RUN


def run(script,*arguments):
    logs=RUN/'execution_logs';logs.mkdir(parents=True,exist_ok=True)
    path=logs/(script.removesuffix('.py')+'_'+'_'.join(arguments)+f'_{time.time_ns()}.log')
    with path.open('x') as stream:
        result=subprocess.run([sys.executable,str(HERE/script),*arguments],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    print(script,arguments,'exit',result.returncode,'log',path,flush=True)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':
    os.environ.update(PYTHONPATH=str(REPO/'verl'),XDG_CACHE_HOME='/tmp/pppo-matched-cache',
                      FLASHINFER_WORKSPACE_BASE='/tmp/pppo-matched-cache/flashinfer',PYTHONDONTWRITEBYTECODE='1')
    try:
        run('prepare.py')
        run('acceptance.py','0');run('acceptance.py','1')
        run('gpu_equivalence.py')
        run('launch.py','smoke-g1');run('launch.py','smoke-g8');run('launch.py','smoke-resume')
        run('finalize.py','gate3')
        run('launch.py','profile');run('finalize.py','gate4')
        run('launch.py','formal')
        run('evaluate_step300.py')
    finally:
        run('finalize.py','reports')
