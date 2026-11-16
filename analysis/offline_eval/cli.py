#!/usr/bin/env python3
"""Portable frozen-protocol evaluation. `submit` is management-only."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from protocol import (CONTEXT, HERE, ORDER, REPO, check_environment, contract, ensure_compute_host,
                      read, reject_overrides, require, selected, source_hashes, state, verifier, write)


@contextmanager
def exclusive(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ('.' + output.name + '.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output


def selections(args):
    args.datasets = selected(args.datasets)
    require(bool(args.horizons) and len(set(args.horizons)) == len(args.horizons), 'Duplicate or empty horizons')
    args.horizons = sorted(args.horizons)
    require(args.step >= 0, '--step must be nonnegative and explicit')
    require(args.gpus > 0, '--gpus must be positive')
    require(args.model_id and '/' not in args.model_id, 'Invalid model ID')


def run(args, check_only=False):
    from data import checkpoint, prepare, tokenizer_identity
    ensure_compute_host()
    selections(args)
    versions = check_environment(generation=not check_only)
    reference = args.tokenizer or args.checkpoint
    prompts = prepare(args.data_dir, reference, args.datasets, args.offline)
    verifier()
    cp = checkpoint(args.checkpoint)
    ti = tokenizer_identity(reference)
    if check_only:
        print(json.dumps(dict(checkpoint=cp, tokenizer=ti, prompts=len(prompts), versions=versions), indent=2))
        return
    from engine import generate, inventory
    from results import score
    devices = inventory(args.gpus, args.devices)
    with exclusive(args.output) as out:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(model_id=args.model_id, step=args.step, checkpoint=cp, tokenizer=ti,
            datasets=args.datasets, horizons=args.horizons, protocol=contract(), source=source_hashes(),
            devices=devices, versions=versions, run_id=uuid.uuid4().hex[:12],
            expected_rows=len(prompts)*32*len(args.horizons), scheduler_job=os.getenv('SLURM_JOB_ID'),
            inherited_cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), created_at=time.time())
        write(out / 'manifest.json', manifest)
        write(out / 'prompts.json', prompts)
        state(out, 'PREPARED', expected_rows=manifest['expected_rows'])
        try:
            generate(out, devices, args.cache_root)
            require(manifest['source'] == source_hashes(), 'Evaluation source changed during generation')
            score(out)
            write(out / 'driver_exit.json', dict(exit_code=0, time=time.time()))
        except BaseException as exc:
            state(out, 'FAILED', error=repr(exc))
            write(out / 'driver_exit.json', dict(exit_code=1, error=repr(exc), time=time.time()))
            raise


def run_arguments(args):
    result = ['run', '--checkpoint', str(args.checkpoint.resolve()), '--model-id', args.model_id,
              '--step', str(args.step), '--output', str(args.output.resolve()),
              '--data-dir', str(args.data_dir.resolve()), '--cache-root', str(args.cache_root),
              '--gpus', str(args.gpus), '--datasets', *args.datasets,
              '--horizons', *map(str, args.horizons), '--offline']
    if args.tokenizer:
        result += ['--tokenizer', str(args.tokenizer.resolve())]
    if args.devices is not None:
        result += ['--devices', *map(str, args.devices)]
    return result


def submission(args):
    from data import download
    selections(args)
    require(args.cpus_per_gpu > 0, '--cpus-per-gpu must be positive')
    limit = {'gpu': 18, 'gpu1': 8}.get(args.partition)
    require(limit is None or args.cpus_per_gpu <= limit, 'CPU request exceeds HPC4 partition limit')
    # Only the four small source files are downloaded here; tokenization/model checks run in allocation.
    download(args.data_dir, args.datasets, args.offline)
    with exclusive(args.output):
        require(not args.output.exists(), 'Evaluation output already exists')
        directory = Path(str(args.output.resolve()) + '.submission')
        directory.mkdir(parents=True, exist_ok=False)
        exports = []
        for key in ('EVAL_CONTAINER', 'EVAL_APPTAINER', 'EVAL_SQUASHFUSE_BIN', 'EVAL_BINDS', 'PYTHON_BIN'):
            if key in os.environ:
                exports.append('export ' + key + '=' + shlex.quote(os.environ[key]))
        launcher = REPO / ('offline_evaluate.sh' if (REPO / 'offline_evaluate.sh').exists() else 'evaluate.sh')
        command = ['bash', str(launcher), *run_arguments(args)]
        script = directory / 'job.sh'
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + '\n'.join(exports) + '\nexec ' + shlex.join(command) + '\n')
        submit = ['yhbatch', '--parsable', '-N', '1', '-p', args.partition,
                  '--gpus-per-node', str(args.gpus), '--cpus-per-gpu', str(args.cpus_per_gpu),
                  '--time', args.time, '--output', str(directory / 'job-%j.out')]
        if args.mem:
            submit += ['--mem', args.mem]
        submit += [str(script)]
        write(directory / 'submission.json', dict(stage='PREPARED', command=submit, run_command=command))
        if args.dry_run:
            print(shlex.join(submit))
            return
        result = subprocess.run(submit, capture_output=True, text=True)
        require(result.returncode == 0, 'Scheduler rejected submission: ' + result.stderr)
        job_id = result.stdout.strip().split(';')[0]
        require(job_id.isdigit(), 'Unexpected scheduler job ID: ' + result.stdout)
        write(directory / 'submission.json', dict(stage='SUBMITTED', job_id=job_id, command=submit, run_command=command))
        print(json.dumps(dict(job_id=job_id, submission=str(directory), output=str(args.output))))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('run', 'check', 'submit'):
        s = sub.add_parser(command)
        s.add_argument('--checkpoint', type=Path, required=True)
        s.add_argument('--tokenizer', type=Path, help='Historical Base tokenizer; defaults to checkpoint assets')
        s.add_argument('--model-id', required=True)
        s.add_argument('--step', type=int, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--data-dir', type=Path, default=Path.home() / '.cache/ncbr-offline-eval/data')
        s.add_argument('--datasets', nargs='+', choices=ORDER, default=list(ORDER))
        s.add_argument('--horizons', nargs='+', type=int, choices=CONTEXT, default=[8192])
        s.add_argument('--gpus', type=int, default=8, help='Desired concurrency; use up to available visible GPUs and eight logical shards')
        s.add_argument('--devices', nargs='+', type=int, help='Logical indices within current CUDA visibility, never host physical IDs')
        s.add_argument('--cache-root', type=Path, default=Path('/tmp/ncbr-eval'))
        s.add_argument('--offline', action='store_true')
        if command == 'submit':
            s.add_argument('--partition', default='gpu1')
            s.add_argument('--cpus-per-gpu', type=int, default=8)
            s.add_argument('--mem')
            s.add_argument('--time', default='24:00:00')
            s.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('prepare-data')
    s.add_argument('--data-dir', type=Path, required=True)
    s.add_argument('--tokenizer', type=Path, required=True)
    s.add_argument('--datasets', nargs='+', choices=ORDER, default=list(ORDER))
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--offline', action='store_true')
    for command in ('status', 'score'):
        s = sub.add_parser(command)
        s.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('compare')
    s.add_argument('--left', type=Path, required=True)
    s.add_argument('--right', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('_worker', help=argparse.SUPPRESS)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--horizon', type=int, choices=CONTEXT, required=True)
    s.add_argument('--shard', type=int, choices=range(8), required=True)
    s.add_argument('--expected-uuid', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    reject_overrides()
    if args.command in ('run', 'check'):
        run(args, args.command == 'check')
    elif args.command == 'submit':
        submission(args)
    elif args.command == 'prepare-data':
        ensure_compute_host()
        check_environment()
        from data import prepare
        require(not args.output.exists(), 'Prepared output already exists')
        prompts = prepare(args.data_dir, args.tokenizer, args.datasets, args.offline)
        write(args.output, prompts)
        print(json.dumps(dict(prompts=len(prompts), output=str(args.output))))
    elif args.command == 'status':
        out = args.output
        info = {p.name: read(p) for p in out.glob('*.json') if p.name in ('status.json', 'driver_exit.json', 'resource_release.json')}
        info['progress'] = {str(p.relative_to(out)): read(p) for p in out.glob('h*/shard*.progress.json')}
        info['submission'] = read(Path(str(out.resolve()) + '.submission') / 'submission.json') if (Path(str(out.resolve()) + '.submission') / 'submission.json').exists() else None
        print(json.dumps(info, indent=2))
    elif args.command == '_worker':
        ensure_compute_host()
        from engine import worker
        worker(args.output, args.horizon, args.shard, args.expected_uuid)
    else:
        ensure_compute_host()
        from results import score, compare
        with exclusive(args.output):
            if args.command == 'score':
                score(args.output)
            else:
                check_environment()
                compare(args.left, args.right, args.output)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
