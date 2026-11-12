#!/usr/bin/env python3
"""Compatibility CLI; generation/scoring are implemented in analysis/offline_eval."""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CORE = HERE / 'offline_eval' if (HERE / 'offline_eval').is_dir() else HERE.parents[1] / 'analysis/offline_eval'
sys.path.insert(0, str(CORE))
import cli
from protocol import lock, require


def bridge(arm, command):
    out = Path(os.environ['EVAL_OUTPUT']) / arm
    if command == 'score':
        return ['score', '--output', str(out)]
    data = Path(os.environ['EVAL_DATA']).parent / 'pinned_inputs'
    data.mkdir(parents=True, exist_ok=True)
    for name, variable in [('AIME2024', 'VAL_FILE_1'), ('AIME2025', 'VAL_FILE_2'), ('AMC23', 'AMC23_FILE')]:
        source = Path(os.environ[variable]).resolve(strict=True)
        target = data / lock()['datasets'][name]['file']
        if not target.exists():
            target.symlink_to(source)
    args = [command, '--checkpoint', os.environ[arm.upper() + '_HF'], '--model-id', arm,
            '--tokenizer', os.environ['MODEL_PATH'], '--step', os.environ['EVAL_STEP'],
            '--output', str(out), '--data-dir', str(data), '--offline',
            '--datasets', 'AIME2024', 'AIME2025', 'AMC23', '--horizons', '2048', '8192']
    # Deprecated EVAL_GPUS means logical indices within existing CUDA visibility.
    if os.getenv('EVAL_GPUS'):
        slots = os.environ['EVAL_GPUS'].split(',')
        require(all(s.strip().isdigit() for s in slots), 'EVAL_GPUS now accepts logical indices; keep UUIDs in scheduler CUDA_VISIBLE_DEVICES')
        args += ['--devices', *slots, '--gpus', str(len(slots))]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-data', 'check', 'run', 'score', 'compare'))
    parser.add_argument('arm', choices=('dapo', 'ncbr', 'all'), nargs='?', default='all')
    args = parser.parse_args(argv)
    if args.command == 'prepare-data':
        inputs = bridge('dapo', 'check')
        data = inputs[inputs.index('--data-dir') + 1]
        return cli.main(['prepare-data', '--data-dir', data, '--tokenizer', os.environ['MODEL_PATH'],
                         '--output', os.environ['EVAL_DATA'], '--datasets', 'AIME2024', 'AIME2025', 'AMC23', '--offline'])
    if args.command != 'compare':
        for arm in ('dapo', 'ncbr') if args.arm == 'all' else (args.arm,):
            cli.main(bridge(arm, args.command))
    if args.command == 'compare' or (args.command == 'run' and args.arm == 'all'):
        out = Path(os.environ['EVAL_OUTPUT'])
        return cli.main(['compare', '--left', str(out/'dapo'), '--right', str(out/'ncbr'), '--output', str(out/'comparison')])
    return 0


if __name__ == '__main__':
    sys.exit(main())
