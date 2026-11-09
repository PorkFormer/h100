#!/usr/bin/env python3
"""Run frozen behavioral goldens and CPU regressions without Ray/GPU initialization."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import re

import run

source = run.source_root()
run.verify_source()
run.verify_recipe()
output = Path(tempfile.mkdtemp(prefix='q4repro-cpu-'))
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           FLASHINFER_WORKSPACE_BASE=str(output / 'flashinfer'), XDG_CACHE_HOME=str(output / 'cache'))
goldens = source / 'tests/experimental/ncbr_portable'
summary = {'source': run.verify_source(), 'goldens': {}}
for script, name in [('characterize.py', 'golden.json'), ('characterize_boundaries.py', 'boundary_golden.json')]:
    with (output / f'{script}.log').open('w') as log:
        subprocess.run([sys.executable, str(goldens / script), '--root', str(source.parent),
                        '--output', str(output / name)], env=env, cwd=output,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    run.require((output / name).read_bytes() == (goldens / name).read_bytes(), f'Golden differs: {name}')
    summary['goldens'][name] = run.sha256(output / name)
paths = [source / 'tests/experimental/natural_continuation_boundary_return',
         goldens / 'test_standard_on_cpu.py', goldens / 'test_hook_on_cpu.py',
         source / 'tests/experimental/probe_credit/test_dapo_trainer_on_cpu.py',
         source / 'tests/experimental/probe_credit/test_dynamic_sampling_on_cpu.py',
         run.ROOT / 'test_reproduction.py', run.ROOT / 'test_offline_eval.py']
args = ['-q', '-p', 'no:cacheprovider', *map(str, paths)]
bootstrap = ('import ray; ray.init=lambda *a,**k: (_ for _ in ()).throw(AssertionError("Ray forbidden")); '
             f'import pytest; raise SystemExit(pytest.main({args!r}))')
with (output / 'pytest.log').open('w') as log:
    subprocess.run([sys.executable, '-c', bootstrap], env=env, cwd=output,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
summary['status'] = 'PASS'
test_log = (output / 'pytest.log').read_text()
summary['tests_passed'] = int(re.search(r'(\d+) passed', test_log).group(1))
run.write_json(output / 'summary.json', summary)
print(f'{summary["tests_passed"]} CPU tests passed; both frozen goldens match exactly.')
print(f'PASS: {output}')
