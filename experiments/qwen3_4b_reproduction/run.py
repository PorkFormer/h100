#!/usr/bin/env python3
"""Portable, fresh-start historical DAPO / NCBR reproduction launcher."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def source_root():
    candidate = os.environ.get('VERL_ROOT')
    root = Path(candidate) if candidate else (ROOT / 'verl' if (ROOT / 'verl').is_dir() else ROOT.parents[1] / 'verl')
    root = root.resolve()
    require((root / 'verl/__init__.py').is_file(), f'Missing verl source: {root}')
    return root


def verify_source():
    root = source_root()
    manifest = read_json(ROOT / 'provenance/source.json')
    for name, digest in manifest['files'].items():
        require((root / name).is_file() and sha256(root / name) == digest, f'Source changed/missing: {root / name}')
    # Extra Python/YAML files can change import resolution or Hydra defaults too.
    runtime_files = {p.relative_to(root).as_posix() for p in (root / 'verl').rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}
    require(not (runtime_files - set(manifest['files'])), 'Unrecorded source/config files in verl runtime')
    return {'commit': manifest['commit'], 'files_verified': len(manifest['files'])}


def verify_recipe():
    manifest = read_json(ROOT / 'provenance/recipe.json')
    for name, digest in manifest.items():
        require(sha256(ROOT / name) == digest, f'Frozen recipe changed: {name}; edit settings.env for machine paths')


def environment():
    require(sys.version_info[:2] == (3, 12), 'Python 3.12 is required')
    expected = read_json(ROOT / 'provenance/dependencies.json')
    actual = {}
    for name, version in expected['versions'].items():
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise ValueError(f'Missing dependency: {name}=={version}') from None
        require(actual[name] == version, f'{name}: expected {version}, got {actual[name]}')
    supplement_status = {}
    for name, version in expected.get('portable_supplements', {}).items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            # The original system interpreter supplies cryptography's CFFI backend via apt.
            # A new isolated venv must have the portable wheel dependencies explicitly.
            require(sys.prefix == sys.base_prefix, f'Missing portable dependency: {name}=={version}')
            import _cffi_backend
            require(Path(_cffi_backend.__file__).is_file(), 'Missing system CFFI backend')
            supplement_status[name] = 'source-system CFFI backend; target venv must install pinned wheel'
        else:
            require(installed == version, f'{name}: expected {version}, got {installed}')
            supplement_status[name] = installed
    sys.path.insert(0, str(source_root()))
    import verl
    require(Path(verl.__file__).resolve().is_relative_to(source_root()), 'Wrong verl imported')
    # Real imports catch binary/API errors that metadata alone cannot catch; no Ray initialization.
    from verl.experimental.natural_continuation_boundary_return.main_dapo_boundary_return import BoundaryReturnTaskRunner
    import flash_attn
    import vllm
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    # Exercises the binary crypto backend used by auth dependencies, without network or credentials.
    Ed25519PrivateKey.generate().public_key()
    del BoundaryReturnTaskRunner, flash_attn, vllm
    return {'python': sys.version, 'versions': actual, 'source': verify_source(),
            'portable_supplements': supplement_status,
            'known_metadata_conflicts': expected['metadata_conflicts']}


def settings():
    result = {}
    for key in ('MODEL_PATH', 'TRAIN_FILE', 'VAL_FILE_1', 'VAL_FILE_2', 'OUTPUT_ROOT', 'CACHE_ROOT'):
        require(os.environ.get(key), f'Set {key} in settings.env')
        result[key] = str(Path(os.environ[key]).expanduser().resolve())
    for key, default in [('GPUS_PER_NODE', 8), ('RAY_CPUS', 32), ('RAY_OBJECT_STORE_GB', 16)]:
        result[key] = int(os.environ.get(key, default))
        require(result[key] > 0, f'{key} must be positive')
    result['RUN_TAG'] = os.environ.get('RUN_TAG', 'repro_v1')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', result['RUN_TAG']), 'RUN_TAG must contain only letters, digits, _ or -')
    result['PROJECT_NAME'] = os.environ.get('PROJECT_NAME', 'qwen3_4b_reproduction')
    return result


def run_dir(s, arm, smoke=False):
    return Path(s['OUTPUT_ROOT']) / f'{arm}_{"smoke_s2" if smoke else "s300"}_{s["RUN_TAG"]}'


def flatten(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f'{prefix}.{key}' if prefix else key))
        return out
    return {prefix: value}


def historical_diff(config, arm):
    from omegaconf import OmegaConf
    old = flatten(OmegaConf.to_container(OmegaConf.load(ROOT / 'reference' / f'{arm}.yaml'), resolve=True))
    new = flatten(OmegaConf.to_container(config, resolve=True))
    allowed = {
        'data.train_files': 'asset location', 'data.val_files': 'asset location',
        'trainer.nnodes': 'single-node topology', 'trainer.n_gpus_per_node': 'GPU topology',
        'trainer.total_training_steps': '300-step target / separate smoke',
        'trainer.save_freq': 'smoke checkpoint only', 'trainer.test_freq': 'smoke validation only',
        'trainer.resume_mode': 'fresh Base initialization', 'trainer.resume_from_path': 'fresh Base initialization',
        'trainer.val_before_train': 'fresh run initial validation',
        'trainer.logger': 'local JSONL + offline W&B',
        'trainer.project_name': 'run identity', 'trainer.experiment_name': 'run identity',
        'trainer.default_local_dir': 'output location',
        'actor_rollout_ref.rollout.trace.project_name': 'run identity',
        'actor_rollout_ref.rollout.trace.experiment_name': 'run identity',
        'actor_rollout_ref.rollout.n_gpus_per_node': 'GPU topology',
    }
    changes = {}
    for key, value in old.items():
        require(key in new, f'Historical field disappeared: {key}')
        if new[key] == value:
            continue
        reason = allowed.get(key)
        if value == '__MODEL_PATH__':
            reason = 'model/tokenizer location (including inactive subsystem placeholders)'
        require(reason is not None, f'Unexpected historical change: {key}: {value!r} -> {new[key]!r}')
        changes[key] = {'historical': value, 'current': new[key], 'reason': reason}
    return {'changed_historical_fields': changes,
            'current_interface_additions': {k: v for k, v in new.items() if k not in old}}


def build_config(s, arm, smoke=False):
    from omegaconf import OmegaConf
    verify_recipe()
    out = run_dir(s, arm, smoke)
    raw = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs' / f'{arm}.yaml'), resolve=True)
    substitutions = {'__MODEL_PATH__': s['MODEL_PATH'], '__TRAIN_FILE__': s['TRAIN_FILE'],
                     '__VAL_FILE_1__': s['VAL_FILE_1'], '__VAL_FILE_2__': s['VAL_FILE_2'],
                     '__CHECKPOINT_DIR__': str(out / 'checkpoints')}

    def replace(value):
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return substitutions.get(value, value) if isinstance(value, str) else value

    c = OmegaConf.create(replace(raw))
    c.trainer.n_gpus_per_node = s['GPUS_PER_NODE']
    c.actor_rollout_ref.rollout.n_gpus_per_node = s['GPUS_PER_NODE']
    c.trainer.project_name = s['PROJECT_NAME']
    c.trainer.experiment_name = out.name
    c.actor_rollout_ref.rollout.trace.project_name = s['PROJECT_NAME']
    c.actor_rollout_ref.rollout.trace.experiment_name = out.name
    if smoke:
        c.trainer.total_training_steps = 2
        c.trainer.save_freq = 2
        c.trainer.test_freq = 2
    c.ray_kwargs.ray_init = {'address': 'local', 'num_cpus': s['RAY_CPUS'],
                             'num_gpus': s['GPUS_PER_NODE'], 'include_dashboard': False,
                             'object_store_memory': s['RAY_OBJECT_STORE_GB'] * (1 << 30)}
    OmegaConf.resolve(c)
    historical_diff(c, arm)
    return c


def validate_config(c):
    from verl.experimental.natural_continuation_boundary_return.validation import validate_boundary_return_preflight
    from verl.utils.config import validate_config as framework_validate, omega_conf_to_dataclass
    n = c.trainer.n_gpus_per_node
    require(c.trainer.nnodes == 1, 'This launcher is single-node')
    require(n % c.actor_rollout_ref.rollout.tensor_model_parallel_size == 0, 'GPUs must be divisible by TP=2')
    require(128 % n == 0, '128-response mini-batch must be divisible by GPU count')
    framework_validate(c, use_reference_policy=False, use_critic=False)
    validate_boundary_return_preflight(c)
    for key in ('actor', 'rollout'):
        omega_conf_to_dataclass(c.actor_rollout_ref[key])


def check_assets(s):
    import pyarrow.parquet as pq
    manifest = read_json(ROOT / 'provenance/assets.json')
    checked = {}
    for name, entry in manifest['model'].items():
        path = Path(s['MODEL_PATH']) / name
        require(path.is_file(), f'Missing Base model file: {path}')
        require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'], f'Base model hash mismatch: {path}')
        checked[str(path)] = entry['sha256']
    for key, entry in manifest['data'].items():
        path = Path(s[key])
        require(path.is_file(), f'Missing original data: {path}')
        require(sha256(path) == entry['sha256'], f'Data hash mismatch: {path}; do not reprocess/deduplicate')
        parquet = pq.ParquetFile(path)
        require(parquet.metadata.num_rows == entry['rows'], f'Wrong row count: {path}')
        require({'prompt', 'reward_model', 'data_source'} <= set(parquet.schema_arrow.names), f'Invalid verl data schema: {path}')
        checked[str(path)] = entry['sha256']
    # Validate tokenizer compatibility on a single row, not the 1.79M-row training set.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(s['MODEL_PATH'], local_files_only=True)
    row = next(pq.ParquetFile(s['TRAIN_FILE']).iter_batches(batch_size=1)).to_pylist()[0]
    require(len(tokenizer.apply_chat_template(row['prompt'], add_generation_prompt=True)) > 0, 'Base chat template failed')
    return checked


def check_gpu(s):
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args], text=True, stderr=subprocess.STDOUT)
    rows = [line.split(', ') for line in query(['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                                              '--format=csv,noheader,nounits']).strip().splitlines()]
    devices = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
    require(len(devices) == s['GPUS_PER_NODE'] and len(set(devices)) == len(devices), 'CUDA_VISIBLE_DEVICES count/duplicates')
    selected = []
    for device in devices:
        matches = [r for r in rows if device in (r[0], r[1])]
        require(len(matches) == 1, f'Unknown/ambiguous GPU: {device}; use numeric indices or full GPU UUIDs')
        row = matches[0]
        require(float(row[3]) >= 76000, f'{row[2]} has less than 80GB-class memory')
        require(float(row[4]) <= 1024 and float(row[5]) == 0, f'GPU busy: {row}')
        selected.append(row)
    require(len({r[1] for r in selected}) == len(devices), 'Duplicate physical GPU aliases')
    apps = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    require(not any(row[1] in apps for row in selected), 'Selected GPUs have compute processes')
    import torch
    require(torch.cuda.device_count() == len(devices), 'PyTorch visible GPU count mismatch')
    for i in range(len(devices)):
        require(torch.cuda.get_device_capability(i)[0] >= 8, 'BF16 requires Ampere or newer')
    return {'devices': selected, 'driver': query(['--query-gpu=driver_version', '--format=csv,noheader']).strip()}


def check_resources(s, arms, smoke):
    import psutil
    root = Path(s['OUTPUT_ROOT'])
    existing = root
    while not existing.exists():
        existing = existing.parent
    require(existing.is_dir() and os.access(existing, os.W_OK | os.X_OK),
            f'Output parent is not a writable directory: {existing}')
    cache = Path(s['CACHE_ROOT'])
    while not cache.exists():
        cache = cache.parent
    require(cache.is_dir() and os.access(cache, os.W_OK | os.X_OK),
            f'Cache parent is not a writable directory: {cache}')
    checkpoints = len(arms) if smoke else sum(3 if a == 'dapo' else 6 for a in arms)
    required = checkpoints * 70_000_000_000 + 40_000_000_000
    free = shutil.disk_usage(existing).free
    require(free >= required, f'Insufficient disk: need approximately {required / 1e9:.0f} GB free, found {free / 1e9:.0f} GB')
    shm = shutil.disk_usage('/dev/shm').free
    require(shm >= (s['RAY_OBJECT_STORE_GB'] + 4) * (1 << 30), 'Insufficient /dev/shm; use container --ipc=host or --shm-size=64g')
    ram = psutil.virtual_memory().available
    require(ram >= 128 * (1 << 30), 'Less than 128 GiB available host RAM; 512 GiB host RAM recommended')
    require((os.cpu_count() or 0) >= s['RAY_CPUS'], 'RAY_CPUS exceeds host logical CPUs')
    return {'disk_free_bytes': free, 'disk_budget_bytes': required, 'shm_free_bytes': shm, 'ram_available_bytes': ram}


def verify_checkpoint(run, step, world_size):
    checkpoint = Path(run) / 'checkpoints' / f'global_step_{step}'
    require((checkpoint / 'data.pt').is_file(), f'Missing dataloader state: {checkpoint}')
    actor = checkpoint / 'actor'
    files = []
    for kind in ('model', 'optim', 'extra_state'):
        for rank in range(world_size):
            path = actor / f'{kind}_world_size_{world_size}_rank_{rank}.pt'
            require(path.is_file() and path.stat().st_size > 0, f'Missing/empty checkpoint shard: {path}')
            files.append(path)
    hf = actor / 'huggingface'
    require((hf / 'config.json').is_file(), f'Missing HF config: {hf}')
    index = hf / 'model.safetensors.index.json'
    weights = sorted(set(read_json(index)['weight_map'].values())) if index.is_file() else ['model.safetensors']
    from safetensors import safe_open
    tensor_count = 0
    for name in weights:
        path = hf / name
        require(path.is_file(), f'Missing HF weights: {path}')
        with safe_open(path, framework='pt', device='cpu') as handle:
            tensor_count += len(list(handle.keys()))
        files.append(path)
    require(tensor_count > 0, 'Empty HF weights')
    return {'step': step, 'files': len(files), 'hf_tensors': tensor_count,
            'bytes': sum(p.stat().st_size for p in files)}


def verify_run(out, arm, steps, world_size, smoke):
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    training = {int(r['step']): r['data'] for r in metrics if 'training/global_step' in r['data']}
    require(set(training) == set(range(1, steps + 1)), f'Incomplete trainer steps: expected 1..{steps}')
    for step, values in training.items():
        require(values['training/global_step'] == step, f'Step metric mismatch: {step}')
        for key, val in values.items():
            if isinstance(val, (int, float)):
                require(math.isfinite(val), f'Non-finite metric at step {step}: {key}')
        for key in ('actor/pg_loss', 'actor/grad_norm'):
            require(isinstance(values.get(key), (int, float)) and math.isfinite(values[key]),
                    f'Missing/non-finite health metric at step {step}: {key}')
    validation_steps = {int(r['step']) for r in metrics if any(k.startswith(('val/', 'val-core/', 'val-aux/')) for k in r['data'])}
    expected_val = {0, steps} | set(range(10, steps + 1, 10))
    require(expected_val <= validation_steps, f'Missing validation steps: {expected_val - validation_steps}')
    request_count = sum(float(v.get('boundary_return/continuation_request_count', 0)) for v in training.values())
    require(request_count > 0 if arm == 'ncbr' else request_count == 0, f'Unexpected continuation count: {request_count}')
    interval = steps if smoke else (100 if arm == 'dapo' else 50)
    checkpoints = [verify_checkpoint(out, step, world_size) for step in range(interval, steps + 1, interval)]
    require(int((out / 'checkpoints/latest_checkpointed_iteration.txt').read_text()) == steps, 'Final checkpoint pointer mismatch')
    return {'state': 'complete', 'steps': steps, 'validation_steps': sorted(validation_steps),
            'continuation_requests': request_count, 'checkpoints': checkpoints}


def child(config_path):
    from omegaconf import OmegaConf
    import ray
    from verl.trainer.main_ppo import run_ppo
    from verl.experimental.natural_continuation_boundary_return.main_dapo_boundary_return import BoundaryReturnTaskRunner
    c = OmegaConf.load(config_path)
    validate_config(c)
    try:
        run_ppo(c, task_runner_class=ray.remote(num_cpus=1)(BoundaryReturnTaskRunner))
    finally:
        ray.shutdown()


def launch(s, arm, smoke, preflight):
    from omegaconf import OmegaConf
    out = run_dir(s, arm, smoke)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'provenance').mkdir()
    (out / 'wandb').mkdir()
    c = build_config(s, arm, smoke)
    ray_tmp = tempfile.mkdtemp(prefix='q4r-', dir='/tmp')
    env = dict(os.environ)
    forwarded = {'WANDB_MODE': 'offline', 'WANDB_RESUME': 'never',
                 'WANDB_RUN_ID': os.urandom(6).hex(),
                 'WANDB_DIR': str(out / 'wandb'), 'WANDB_NAME': out.name,
                 'WANDB_PROJECT': s['PROJECT_NAME'], 'VERL_FILE_LOGGER_PATH': str(out / 'metrics.jsonl'),
                 'PYTHONPATH': str(source_root()), 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                 'PYTHONDONTWRITEBYTECODE': '1', 'RAY_USAGE_STATS_ENABLED': '0',
                 'XDG_CACHE_HOME': str(Path(s['CACHE_ROOT']) / 'xdg'),
                 'FLASHINFER_WORKSPACE_BASE': str(Path(s['CACHE_ROOT']) / 'flashinfer'),
                 'TORCH_EXTENSIONS_DIR': str(Path(s['CACHE_ROOT']) / 'torch_extensions'),
                 'TRITON_CACHE_DIR': str(Path(s['CACHE_ROOT']) / 'triton')}
    env.update(forwarded)
    c.ray_kwargs.ray_init._temp_dir = ray_tmp
    c.ray_kwargs.ray_init.runtime_env = {'env_vars': forwarded}
    config_path = out / 'provenance/resolved_config.yaml'
    OmegaConf.save(c, config_path, resolve=True)
    write_json(out / 'provenance/historical_diff.json', historical_diff(c, arm))
    write_json(out / 'provenance/preflight.json', preflight)
    command = [sys.executable, str(ROOT / 'run.py'), '_child', str(config_path)]
    write_json(out / 'provenance/command.json', command)
    write_json(out / 'status.json', {'state': 'running', 'started_at': time.time(), 'arm': arm, 'smoke': smoke})
    try:
        with (out / 'train.log').open('x') as log:
            # Foreground driver; Ray owns only this local runtime. Never call global ray stop.
            process = subprocess.Popen(command, cwd=source_root(), env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
                require(process.wait() == 0, f'Training failed; inspect {out / "train.log"}')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        receipt = verify_run(out, arm, 2 if smoke else 300, s['GPUS_PER_NODE'], smoke)
        write_json(out / 'status.json', receipt)
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': str(error), 'time': time.time()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('environment', 'config', 'check', 'smoke', 'train', '_child'))
    parser.add_argument('arm', nargs='?', default='all')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(source_root()))
    if args.command == '_child':
        child(args.arm)
        return
    require(args.arm in ('all', 'dapo', 'ncbr'), 'Choose dapo, ncbr or all')
    verify_recipe()
    if args.command == 'environment':
        print(json.dumps(environment(), indent=2))
        return
    s = settings()
    arms = ['dapo', 'ncbr'] if args.arm == 'all' else [args.arm]
    smoke = args.command == 'smoke'
    for arm in arms:
        c = build_config(s, arm, smoke)
        validate_config(c)
        print(f'PASS config: {arm}, context={c.actor_rollout_ref.rollout.max_model_len}, fresh Base, GPUs={s["GPUS_PER_NODE"]}')
    if args.command == 'config':
        print('CPU configuration validation only; no assets, GPU checks, Ray, or training.')
        return
    preflight = {'environment': environment(), 'assets': check_assets(s)}
    for arm in arms:
        require(not run_dir(s, arm, smoke).exists(), f'Output already exists: {run_dir(s, arm, smoke)}; choose a new RUN_TAG')
    preflight['resources'] = check_resources(s, arms, smoke)
    # Parent must not initialize CUDA before a child reserves the GPUs.
    gpu_command = [sys.executable, '-c', 'import run; print(__import__("json").dumps(run.check_gpu(run.settings())))']
    def gpu_probe():
        result = subprocess.run(gpu_command, cwd=ROOT, capture_output=True, text=True)
        require(result.returncode == 0, result.stderr or result.stdout)
        return json.loads(result.stdout)
    preflight['gpu'] = gpu_probe()
    preflight['checked_at'] = time.time()
    if args.command == 'check':
        print(json.dumps(preflight, indent=2))
        print('PASS preflight; no Ray or training started.')
        return
    # Prevent two copies of this launcher using the same local GPU set, even across output roots.
    lock_handles = []
    for uuid in sorted(r[1] for r in preflight['gpu']['devices']):
        handle = open(f'/tmp/q4repro-{uuid}.lock', 'a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_handles.append(handle)
    for arm in arms:
        preflight['gpu'] = gpu_probe()
        preflight['checked_at'] = time.time()
        launch(s, arm, smoke, preflight)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
