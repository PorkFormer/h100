"""Start the finite evaluation job detached; never monitor or resume training."""
import os
from pathlib import Path
import subprocess
import sys
import time

import evaluate as e


def main():
    e.require(e.read(e.RUN / 'pause_step20_receipt.json')['status'] == 'PAUSED_AT_STEP20',
              'Training must be paused before starting evaluation')
    e.require(e.read(e.RUN / 'training_status.json')['state'] == 'PAUSED_AT_STEP20',
              'Training status does not confirm pause')
    e.require(all(r['status'] == 'PASS' for r in e.read(e.OUT / 'checkpoint_preflight.json').values()),
              'All checkpoints must pass preflight')
    e.inputs('pppo_s20')
    e.idle_head()
    e.require(not (e.OUT / 'launch_receipt.json').exists(), 'Existing evaluation launch; inspect before retry')
    with (e.OUT / 'evaluation.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('execute.py'))],
                                 cwd=e.REPO, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'PYTHONPATH': str(e.REPO / 'verl'),
                                      'PYTHONDONTWRITEBYTECODE': '1',
                                      'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
    receipt = dict(pid=child.pid, started_at=time.time(), command='execute.py',
                   code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=e.REPO, text=True).strip(),
                   training_resume=False, monitoring_daemon=False,
                   runtime_environment={'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
    e.write(e.OUT / 'launch_receipt.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
