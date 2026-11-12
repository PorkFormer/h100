"""The public evaluation contract. Runtime resource knobs live in cli.py."""
import hashlib
import importlib.util
import json
import os
import socket
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / 'verl').is_dir() else HERE.parents[1]
ORDER = ('AIME2024', 'AIME2025', 'AMC23', 'MATH500')
FIELDS = ('prompt_id', 'benchmark', 'canonical_prompt', 'rendered_prompt', 'prompt_token_ids', 'ground_truth')
CONTEXT = {2048: 4096, 8192: 9216, 16384: 17408}
ENGINE = dict(tensor_parallel_size=1, dtype='bfloat16', gpu_memory_utilization=0.72,
              max_num_seqs=8, max_num_batched_tokens=32768, enable_chunked_prefill=True,
              enable_prefix_caching=True, trust_remote_code=False, seed=42, disable_log_stats=True,
              generation_config='vllm')
VERSIONS = {'vllm': '0.18.0', 'torch': '2.10.0+cu129', 'transformers': '5.3.0',
            'numpy': '1.26.4', 'pyarrow': '23.0.1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(32 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def lock():
    return read(HERE / 'protocol.lock.json')


def contract():
    return dict(version='offline-eval-v2', assets=lock(), samples=32, logical_shards=8,
                batch_size=8, master_seed=42, seed_rule='offline-answer-timing-v1',
                sampling=sampling(8192, 0), contexts={str(k): v for k, v in CONTEXT.items()}, engine=ENGINE, versions=VERSIONS,
                bootstrap_replicates=10000, bootstrap_seed=20260812, input_limit=1024)


def seed(prompt_id, rollout):
    payload = f'offline-answer-timing-v1\0{42}\0{prompt_id}\0{rollout}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little') & 0x7fffffff


def sampling(horizon, sample_seed):
    require(horizon in CONTEXT, 'Unknown frozen horizon')
    return dict(n=1, temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                max_tokens=horizon, ignore_eos=False, seed=sample_seed)


def jobs(prompts, shard):
    return [(p, r) for i, p in enumerate(prompts) for r in range(32) if (i * 32 + r) % 8 == shard]


def selected(names):
    require(bool(names) and len(names) == len(set(names)) and set(names) <= set(ORDER), 'Invalid dataset selection')
    return [b for b in ORDER if b in names]


def validate_prompts(prompts, names):
    specs = lock()['datasets']
    expected = []
    for name in selected(names):
        group = [p for p in prompts if p['benchmark'] == name]
        s = specs[name]
        require(len(group) == s['count'], 'Wrong question count: ' + name)
        require([p['prompt_id'] for p in group] == list(range(s['first_id'], s['first_id'] + s['count'])), 'Question ID/order mismatch')
        require(digest([{k: p[k] for k in FIELDS} for p in group]) == s['semantic_sha256'], 'Frozen prompt/answer/tokenizer mismatch: ' + name)
        require(all(0 < len(p['prompt_token_ids']) <= 1024 for p in group), 'Prompt exceeds frozen input limit')
        expected.extend(group)
    require(prompts == expected, 'Unexpected questions or dataset order')
    return prompts


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def verifier():
    path = REPO / 'verl/verl/utils/reward_score/math_dapo.py'
    require(sha(path) == lock()['verifier_sha256'], 'Frozen scorer changed')
    spec = importlib.util.spec_from_file_location('offline_eval_frozen_math_dapo', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def source_hashes():
    return {p.name: sha(p) for p in sorted(HERE.iterdir()) if (p.suffix == '.py' or p.name == 'protocol.lock.json') and not p.name.startswith('test_')}


def ensure_compute_host():
    require(socket.gethostname().split('.')[0] not in ('th-hpc4-ln0', 'th-hpc4-ln1'),
            'Compute operation on HPC4 login node: use evaluate.sh submit or a CPU scheduler allocation')


def check_environment(generation=False):
    import importlib.metadata as metadata
    wanted = VERSIONS if generation else {k: VERSIONS[k] for k in ('numpy', 'pyarrow', 'transformers')}
    actual = {k: metadata.version(k) for k in wanted}
    require(actual == wanted, 'Frozen software mismatch: ' + canonical(dict(expected=wanted, actual=actual)))
    return actual


def reject_overrides():
    names = ('TEMPERATURE', 'TOP_P', 'TOP_K', 'SAMPLES_PER_PROMPT', 'MASTER_SEED',
             'BATCH_SIZE', 'DTYPE', 'TENSOR_PARALLEL_SIZE', 'GPU_MEMORY_UTILIZATION',
             'MAX_NUM_SEQS', 'MAX_NUM_BATCHED_TOKENS', 'IGNORE_EOS', 'REPETITION_PENALTY')
    bad = [k for k in os.environ if k.startswith('EVAL_') and k[5:] in names]
    require(not bad, 'Frozen evaluation settings cannot be overridden: ' + ', '.join(bad))


def state(out, stage, **extra):
    write(Path(out) / 'status.json', dict(stage=stage, updated_at=time.time(), **extra))
