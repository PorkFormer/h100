"""Frozen identities and fail-closed row validation for AMC23 evaluation."""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HIST = ROOT.parent / 'ncbr_boundary_return_s300_seed1_20260828'
VERL = Path('/workspace/rl/h100-natural-continuation-boundary-return-v1/verl')
TOKENIZER = '/workspace/models/Qwen3-4B-Base'
REVISION = '80815d37005feb82cd7f8fbc6901d5d3eff43057'
DATA_SHA = 'b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e'
DATA_URL = f'https://hf-mirror.com/datasets/math-ai/amc23/resolve/{REVISION}/test-00000-of-00001.parquet'
HORIZONS = (2048, 8192)
SAMPLES = 32
BOOTSTRAP_SEED = 20260812

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(32 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(tmp, path)

def integer_answer(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'non-numeric AMC answer: {value!r}')
    if not math.isfinite(value) or int(value) != value:
        raise ValueError(f'non-integral AMC answer: {value!r}')
    return str(int(value))

def stable_sampling_seed(master_seed, prompt_id, rollout_index):
    payload = f'offline-answer-timing-v1\0{master_seed}\0{prompt_id}\0{rollout_index}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little') & 0x7FFFFFFF

def load_protocol():
    return json.loads((ROOT / 'generation_config.json').read_text())

def models(smoke=False):
    all_models = json.loads((ROOT / 'checkpoint_manifest.json').read_text())['checkpoints']
    return all_models

def prompts(smoke=False):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / 'prompts.parquet').to_pylist()
    return [r for b in ('AIME2024', 'AIME2025', 'AMC23') for r in [x for x in rows if x['benchmark'] == b][:2]] if smoke else rows

def output_root(smoke=False):
    return ROOT / ('smoke' if smoke else 'formal')

def raw_path(horizon, smoke=False):
    if horizon not in HORIZONS:
        raise ValueError('unsupported horizon')
    return output_root(smoke) / f'generation/h{horizon}/raw_generations.jsonl'

def expected_keys(smoke=False):
    return {(m['model_id'], p['prompt_id'], i)
            for m in models(smoke) for p in prompts(smoke) for i in range(SAMPLES)}

def check_rows(rows, horizon, model_rows, prompt_rows, complete=False):
    mm = {m['model_id']: m for m in model_rows}
    pp = {p['prompt_id']: p for p in prompt_rows}
    expected = {(m, p, i) for m in mm for p in pp for i in range(SAMPLES)}
    seen = set()
    for row in rows:
        key = (row['model_id'], row['prompt_id'], row['rollout_index'])
        if key not in expected or key in seen:
            raise ValueError(f'unexpected or duplicate key: {key}')
        seen.add(key)
        m, p = mm[key[0]], pp[key[1]]
        pair_key = f'{key[0]}|{key[1]}|{key[2]}'
        required = {'horizon': horizon, 'benchmark': p['benchmark'], 'data_source': p['data_source'],
                    'arm': m['arm'], 'step': m['step'], 'checkpoint_path': m['checkpoint_path'],
                    'checkpoint_sha256': m['sha256'], 'prompt_hash': p['prompt_hash'],
                    'prompt_token_ids': p['prompt_token_ids'], 'ground_truth': p['ground_truth'],
                    'original_dataset_index': p['original_dataset_index'],
                    'extra_info_json': p['extra_info_json'],
                    'stable_seed': stable_sampling_seed(42, key[1], key[2]),
                    'pair_key': pair_key, 'trajectory_id': f'{pair_key}|h{horizon}',
                    'request_id': 'offline-' + hashlib.sha256((pair_key + f'|h{horizon}').encode()).hexdigest()[:24]}
        for field, value in required.items():
            if row.get(field) != value:
                raise ValueError(f'identity mismatch {key}: {field}')
        tokens = row.get('response_token_ids')
        if not isinstance(tokens, list) or not 0 < len(tokens) <= horizon:
            raise ValueError(f'invalid token count {key}')
        if any(type(t) is not int or not 0 <= t < 151936 for t in tokens):
            raise ValueError(f'invalid token IDs {key}')
        if row.get('response_token_count') != len(tokens):
            raise ValueError(f'length mismatch {key}')
        if row.get('finish_reason') not in ('stop', 'length'):
            raise ValueError(f'invalid finish reason {key}')
        if row['finish_reason'] == 'length' and len(tokens) != horizon:
            raise ValueError(f'early length termination {key}')
        if row.get('hit_cap') != (row['finish_reason'] == 'length' or len(tokens) >= horizon):
            raise ValueError(f'cap mismatch {key}')
        if not row.get('scoring_error') and (type(row.get('reward_acc')) is not bool or row.get('reward_score') != (1.0 if row['reward_acc'] else -1.0)):
            raise ValueError(f'invalid reward {key}')
    if complete and seen != expected:
        raise ValueError(f'coverage mismatch: {len(seen)}/{len(expected)}')
    return seen

def read_rows(horizon, smoke=False, complete=False):
    rows = []
    path = raw_path(horizon, smoke)
    if path.exists():
        with path.open() as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            for n, line in enumerate(f, 1):
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f'invalid JSONL physical line {n}') from e
    check_rows(rows, horizon, models(smoke), prompts(smoke), complete)
    return rows

def verify_frozen(full_weights=False):
    manifest = json.loads((ROOT / 'freeze.json').read_text())
    for item in manifest['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'frozen file changed: {item["path"]}')
    for m in models():
        for w in m['weight_files']:
            p = Path(w['path'])
            if p.stat().st_size != w['size'] or p.stat().st_mtime_ns != w['mtime_ns']:
                raise ValueError(f'weight metadata changed: {p}')
            if full_weights and sha(p) != w['sha256']:
                raise ValueError(f'weight checksum changed: {p}')
    return True


def pending_requests(requests, existing, model_id):
    """Resume complete eight-request batches; reject an interrupted partial batch."""
    if len(requests) % 8:
        raise ValueError('historical eight-request batch boundary mismatch')
    pending = []
    for start in range(0, len(requests), 8):
        batch = requests[start:start + 8]
        present = [(model_id, int(p['prompt_id']), i) in existing for p, i, _ in batch]
        if any(present) and not all(present):
            raise ValueError('incomplete persisted batch; stop without changing batch composition')
        if not any(present):
            pending.extend(batch)
    return pending
