"""Single-node, Ray-free scheduling of eight fixed logical shards."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from data import checkpoint, tokenizer_identity
from protocol import (CONTEXT, ENGINE, HERE, canonical, check_environment, contract, jobs, read,
                      require, sampling, seed, sha, source_hashes, state, tokenizer, validate_prompts, write)


def uuid_key(value):
    return str(value).lower().removeprefix('gpu-').replace('-', '')


def select_devices(visible, count=8, indices=None):
    require(count > 0, '--gpus must be positive')
    slots = list(range(len(visible))) if indices is None else indices
    require(bool(slots) and len(slots) == len(set(slots)) and all(0 <= i < len(visible) for i in slots),
            '--devices must be unique logical indices within scheduler-visible GPUs')
    return [visible[i] for i in slots[:min(count, 8)]]


def inventory(count, indices):
    import torch
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {uuid_key(uuid.strip()): index.strip() for index, uuid in (line.split(',') for line in query.splitlines())}
    visible = []
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    selectors = inherited.split(',') if inherited else None
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        uuid = str(props.uuid)
        require(uuid_key(uuid) in mapping, 'Cannot safely map visible GPU UUID to a numeric CUDA identifier')
        if selectors is None:
            numeric = str(i)
        elif i < len(selectors) and selectors[i].strip().isdigit():
            numeric = selectors[i].strip()  # Preserve the scheduler's CUDA numbering exactly.
        else:
            numeric = mapping[uuid_key(uuid)]
        visible.append(dict(logical_index=i, numeric_id=numeric, uuid=uuid,
                            name=props.name, memory=props.total_memory, capability=[props.major, props.minor]))
    return select_devices(visible, count, indices)


def worker(out, horizon, shard, expected_uuid):
    import torch
    require(torch.cuda.device_count() == 1, 'Worker must see exactly one assigned GPU')
    require(uuid_key(torch.cuda.get_device_properties(0).uuid) == uuid_key(expected_uuid),
            'Numeric GPU mapping differs from scheduler allocation; refusing to run')
    check_environment(generation=True)
    import vllm
    manifest = read(out / 'manifest.json')
    require(manifest['protocol'] == contract() and manifest['source'] == source_hashes(), 'Frozen protocol/source changed')
    require(tokenizer_identity(manifest['tokenizer']['path']) == manifest['tokenizer'], 'Tokenizer changed')
    prompts = validate_prompts(read(out / 'prompts.json'), manifest['datasets'])
    require(horizon in manifest['horizons'], 'Unselected horizon')
    tok = tokenizer(manifest['tokenizer']['path'])
    llm = vllm.LLM(model=manifest['checkpoint']['path'], tokenizer=manifest['tokenizer']['path'],
                   max_model_len=CONTEXT[horizon], **ENGINE)
    requests = jobs(prompts, shard)
    with (out / f'h{horizon}/shard{shard}.jsonl').open('x') as stream:
        for start in range(0, len(requests), 8):
            batch = requests[start:start + 8]
            outputs = llm.generate([dict(prompt_token_ids=p['prompt_token_ids']) for p, _ in batch],
                [vllm.SamplingParams(**sampling(horizon, seed(p['prompt_id'], r))) for p, r in batch], use_tqdm=False)
            require(len(outputs) == len(batch), 'Incomplete generated batch')
            for (p, r), output in zip(batch, outputs, strict=True):
                require(len(output.outputs) == 1, 'Expected one completion per seeded request')
                result = output.outputs[0]
                ids = list(result.token_ids)
                require(result.finish_reason in ('stop', 'length'), 'Incomplete/aborted generation')
                stream.write(canonical(dict(prompt_id=p['prompt_id'], rollout=r, horizon=horizon, shard=shard,
                    seed=seed(p['prompt_id'], r), checkpoint_sha256=manifest['checkpoint']['sha256'],
                    response_token_ids=ids, response_text=tok.decode(ids, skip_special_tokens=True),
                    finish_reason=result.finish_reason)) + '\n')
            stream.flush()
            write(out / f'h{horizon}/shard{shard}.progress.json', dict(completed=start+len(batch), expected=len(requests), updated_at=time.time()))
    write(out / f'h{horizon}/shard{shard}.receipt.json', dict(rows=len(requests), gpu_uuid=expected_uuid,
        sha256=sha(out / f'h{horizon}/shard{shard}.jsonl')))


def stop_children(children):
    for proc in children:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in children:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        # A successful driver can leave engine subprocesses in its own session.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def generate(out, devices, cache_root):
    manifest = read(out / 'manifest.json')
    children = []
    exit_codes = []
    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for horizon in manifest['horizons']:
            (out / f'h{horizon}').mkdir()
            for start in range(0, 8, len(devices)):
                children = []
                for shard, device in zip(range(start, min(start + len(devices), 8)), devices):
                    cache = Path(cache_root) / f'{manifest["run_id"]}/h{horizon}s{shard}'
                    cache.mkdir(parents=True, exist_ok=False)
                    env = dict(os.environ)
                    # Environment knobs must not silently replace the frozen engine recipe.
                    for key in list(env):
                        if key.startswith(('VLLM_', 'RAY_')) or key in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
                            del env[key]
                    env.update(CUDA_VISIBLE_DEVICES=device['numeric_id'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                               VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                               TMPDIR=str(cache), XDG_CACHE_HOME=str(cache / 'xdg'), TRITON_CACHE_DIR=str(cache / 'triton'),
                               TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), VLLM_CACHE_ROOT=str(cache / 'vllm'))
                    with (out / f'h{horizon}/shard{shard}.log').open('x') as log:
                        children.append(subprocess.Popen([sys.executable, str(HERE / 'cli.py'), '_worker',
                            '--output', str(out), '--horizon', str(horizon), '--shard', str(shard),
                            '--expected-uuid', device['uuid']], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                state(out, 'GENERATING', horizon=horizon, active_shards=list(range(start, min(start+len(devices), 8))))
                try:
                    while any(p.poll() is None for p in children):
                        require(all(p.poll() in (None, 0) for p in children), 'Worker failed; see shard logs')
                        time.sleep(1)
                    require(all(p.returncode == 0 for p in children), 'Worker failed; see shard logs')
                finally:
                    stop_children(children)
                    exit_codes.extend(p.returncode for p in children)
        require(checkpoint(manifest['checkpoint']['path']) == manifest['checkpoint'], 'Checkpoint changed during generation')
        raw = {}
        expected = len(read(out / 'prompts.json')) * 4
        for horizon in manifest['horizons']:
            for shard in range(8):
                relative = f'h{horizon}/shard{shard}.jsonl'
                receipt = read(out / f'h{horizon}/shard{shard}.receipt.json')
                require(receipt['rows'] == expected and receipt['sha256'] == sha(out / relative), 'Worker receipt mismatch')
                raw[relative] = receipt['sha256']
        write(out / 'generation_complete.json', dict(raw_sha256=raw, worker_exit_codes=exit_codes))
    finally:
        stop_children(children)
        write(out / 'resource_release.json', dict(child_processes_exited=all(p.poll() is not None for p in children),
                                                  scope='only this run process groups', worker_exit_codes=exit_codes))
        for sig, handler in previous.items():
            signal.signal(sig, handler)
