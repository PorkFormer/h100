"""CPU regression tests; no Ray initialization or model inference."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli
import data
import engine
import protocol as p
import results as r


def test_defaults_and_fixed_sampling():
    args = cli.parser().parse_args(['run', '--checkpoint', '/model', '--model-id', 'x', '--step', '300', '--output', '/out'])
    assert args.datasets == list(p.ORDER) and args.horizons == [8192] and args.gpus == 8
    assert p.ENGINE['generation_config'] == 'vllm'  # Never inherit model-specific stop/penalty defaults.
    assert p.seed(512, 0) == 239391688
    assert p.seed(512, 1) == 1808425106
    for h in p.CONTEXT:
        assert p.sampling(h, 123) == dict(n=1, temperature=1., top_p=1., top_k=-1,
            repetition_penalty=1., max_tokens=h, ignore_eos=False, seed=123)


@pytest.mark.parametrize('option', ['--temperature', '--samples', '--tp', '--batch-size', '--dtype', '--gpu-memory-utilization'])
def test_no_protocol_overrides(option):
    with pytest.raises(SystemExit):
        cli.parser().parse_args(['run', '--checkpoint', '/m', '--model-id', 'x', '--step', '0', '--output', '/o', option, '1'])


def test_reject_environment_override(monkeypatch):
    monkeypatch.setenv('EVAL_TEMPERATURE', '0')
    with pytest.raises(ValueError, match='Frozen'):
        p.reject_overrides()


@pytest.mark.parametrize('names', [list(p.ORDER), ['AIME2025'], ['AMC23', 'MATH500'], ['MATH500']])
def test_shards_and_subset_seed_stability(names):
    prompts = [dict(prompt_id=i) for b in names for i in range(p.lock()['datasets'][b]['first_id'], p.lock()['datasets'][b]['first_id']+p.lock()['datasets'][b]['count'])]
    coverage = []
    for shard in range(8):
        jobs = p.jobs(prompts, shard)
        assert len(jobs) % 8 == 0
        coverage.extend((q['prompt_id'], roll) for q, roll in jobs)
    assert len(coverage) == len(set(coverage)) == len(prompts)*32
    assert p.seed(1000, 0) == 1682806218


@pytest.mark.parametrize('count', [1, 2, 4, 8, 16])
def test_hardware_is_not_a_gate(count):
    visible = [dict(name='Other GPU', uuid=str(i)) for i in range(count)]
    assert len(engine.select_devices(visible)) == min(count, 8)
    assert engine.select_devices(visible, count=1) == [visible[0]]


def test_visible_subset():
    assert engine.select_devices(['allocated-7', 'allocated-2', 'allocated-5'], 8, [2, 0]) == ['allocated-5', 'allocated-7']
    with pytest.raises(ValueError):
        engine.select_devices(['only-this'], 8, [1])
    with pytest.raises(ValueError):
        engine.select_devices([], 8)
    assert engine.uuid_key('GPU-ABC-123') == engine.uuid_key('abc123')


def test_uuid_inventory_respects_allocation(monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
        get_device_properties=lambda i: SimpleNamespace(uuid=['GPU-BB', 'GPU-DD'][i], name='Other', total_memory=1, major=9, minor=0)))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-BB,GPU-DD')
    monkeypatch.setattr(engine.subprocess, 'check_output', lambda *a, **k: '0, GPU-AA\n1, GPU-BB\n2, GPU-CC\n3, GPU-DD\n')
    assert [d['numeric_id'] for d in engine.inventory(8, None)] == ['1', '3']
    assert [d['numeric_id'] for d in engine.inventory(8, [1])] == ['3']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,2')
    assert [d['numeric_id'] for d in engine.inventory(8, None)] == ['7', '2']
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert [d['numeric_id'] for d in engine.inventory(8, None)] == ['0', '1']


def test_wrong_gpu_uuid_fails_before_vllm(monkeypatch, tmp_path):
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(uuid='GPU-OTHER')))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    with pytest.raises(ValueError, match='GPU mapping'):
        engine.worker(tmp_path, 8192, 0, 'GPU-EXPECTED')


def test_changed_tokenizer_not_silently_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'download', lambda *a, **k: tmp_path)
    with pytest.raises(ValueError, match='tokenizer asset mismatch'):
        data.prepare(tmp_path, tmp_path, ['MATH500'], True)


def test_nonzero_child_is_detected_and_cleaned():
    import subprocess
    process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(3)'], start_new_session=True)
    process.wait()
    engine.stop_children([process])
    assert process.returncode == 3


def test_worker_failure_propagates_and_releases_owned_children(tmp_path, monkeypatch):
    out = tmp_path/'run'
    p.write(out/'manifest.json', {'horizons': [8192], 'run_id': 'test'})
    fake = SimpleNamespace(pid=123456789, returncode=3, poll=lambda: 3)
    cleaned = []
    monkeypatch.setattr(engine.subprocess, 'Popen', lambda *a, **k: fake)
    monkeypatch.setattr(engine, 'stop_children', lambda children: cleaned.extend(children))
    with pytest.raises(ValueError, match='Worker failed'):
        engine.generate(out, [{'numeric_id': '3', 'uuid': 'GPU-allocated'}], tmp_path/'cache')
    assert cleaned and p.read(out/'resource_release.json')['worker_exit_codes'] == [3]


class Tokenizer:
    def decode(self, ids, **kwargs):
        return 'Answer: 42' if ids == [1] else 'Answer: 0'


def row(rollout=0, **changes):
    value = dict(prompt_id=512, rollout=rollout, horizon=2048, checkpoint_sha256='test',
                 seed=p.seed(512, rollout), response_token_ids=[1], response_text='Answer: 42', finish_reason='stop')
    value.update(changes)
    return value


@pytest.mark.parametrize('changes', [dict(rollout=32), dict(seed=1), dict(horizon=8192),
    dict(checkpoint_sha256='changed'), dict(response_token_ids=[-1]), dict(response_token_ids=[1]*2049),
    dict(response_text='changed'), dict(finish_reason='abort')])
def test_bad_trajectory(changes):
    with pytest.raises(ValueError):
        r.validate_row(row(**changes), {'prompt_id': 512}, 2048, 'test', Tokenizer())


def test_full_scoring_and_question_bootstrap(tmp_path):
    prompts = [dict(prompt_id=512, benchmark='AIME2024', ground_truth='42')]
    rows = [row(i, response_token_ids=[1] if i == 0 else [2], response_text='Answer: 42' if i == 0 else 'Answer: 0') for i in range(32)]
    metrics, questions, scored = r.calculate(prompts, [2048], 'test', rows, Tokenizer(), p.verifier(), tmp_path/'audit.json')
    assert metrics[0]['pass_at_1'] == 1/32
    assert metrics[0]['pass_at_32'] == 1
    assert metrics[0]['pass_at_1_ci_low'] == 1/32
    assert len(scored) == 32 and len(questions) == 1
    assert r.interval([0.25]*30) == [0.25, 0.25]


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'wrong_seed', 'verifier_error'])
def test_failure_never_becomes_wrong_answer(tmp_path, kind):
    prompts = [dict(prompt_id=512, benchmark='AIME2024', ground_truth='42')]
    rows = [row(i) for i in range(32)]
    scorer = p.verifier()
    if kind == 'missing':
        rows.pop()
    elif kind == 'duplicate':
        rows.append(rows[0])
    elif kind == 'wrong_seed':
        rows[0]['seed'] = 1
    else:
        def scorer(**kwargs):
            raise RuntimeError('synthetic verifier failure')
    with pytest.raises(ValueError):
        r.calculate(prompts, [2048], 'test', rows, Tokenizer(), scorer, tmp_path/'audit.json')
    if kind == 'verifier_error':
        assert p.read(tmp_path/'audit.json')['verifier_errors'] == 32


def test_failed_rescore_revokes_old_results(tmp_path):
    p.write(tmp_path/'manifest.json', {'protocol': p.contract(), 'datasets': ['AIME2024']})
    p.write(tmp_path/'generation_complete.json', {})
    p.write(tmp_path/'status.json', {'stage': 'COMPLETE'})
    p.write(tmp_path/'scores.json', {'stale': True})
    p.write(tmp_path/'scoring_receipt.json', {'passed': True})
    with pytest.raises(FileNotFoundError):
        r.score(tmp_path)
    assert p.read(tmp_path/'status.json')['stage'] == 'FAILED'
    assert not (tmp_path/'scores.json').exists()
    assert not (tmp_path/'scoring_receipt.json').exists()
    assert list(tmp_path.glob('scoring_attempts/*/previous_scores.json'))


def test_protocol_drift_also_revokes_old_success(tmp_path):
    p.write(tmp_path/'manifest.json', {'protocol': {'changed': True}})
    p.write(tmp_path/'scoring_receipt.json', {'passed': True})
    with pytest.raises(ValueError, match='Protocol changed'):
        r.score(tmp_path)
    assert not (tmp_path/'scoring_receipt.json').exists()
    assert p.read(tmp_path/'status.json')['stage'] == 'FAILED'


def test_run_lock(tmp_path):
    with cli.exclusive(tmp_path/'out'):
        with pytest.raises(BlockingIOError):
            with cli.exclusive(tmp_path/'out'):
                pass


def test_corrupt_local_data_not_redownloaded(tmp_path):
    (tmp_path/'math-500-verl.parquet').write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash mismatch'):
        data.download(tmp_path, ['MATH500'], offline=True)


def test_missing_offline_data(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        data.download(tmp_path, ['MATH500'], offline=True)


def test_submit_preserves_selected_scope_and_quotes(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'download', lambda *a, **k: None)
    output = tmp_path/'has spaces;literal'
    args = cli.parser().parse_args(['submit', '--checkpoint', '/model', '--model-id', 'x', '--step', '300',
        '--output', str(output), '--datasets', 'MATH500', '--horizons', '16384', '--gpus', '2', '--dry-run'])
    cli.submission(args)
    directory = Path(str(output)+'.submission')
    receipt = p.read(directory/'submission.json')
    assert receipt['run_command'][receipt['run_command'].index('--datasets')+1] == 'MATH500'
    assert '--offline' in receipt['run_command']
    assert 'has spaces;literal' in (directory/'job.sh').read_text()
    assert receipt['command'][receipt['command'].index('--gpus-per-node')+1] == '2'


def test_cpu_partition_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'download', lambda *a, **k: None)
    args = cli.parser().parse_args(['submit', '--checkpoint', '/m', '--model-id', 'x', '--step', '0',
        '--output', str(tmp_path/'out'), '--partition', 'gpu1', '--cpus-per-gpu', '9'])
    with pytest.raises(ValueError, match='CPU request'):
        cli.submission(args)


@pytest.mark.parametrize('size,nested_rope', [(4, False), (8, True)])
def test_checkpoint_dimensions_not_reference_model(tmp_path, size, nested_rope):
    import numpy as np
    from safetensors.numpy import save_file
    cfg = dict(model_type='qwen3', hidden_size=size, vocab_size=10, num_hidden_layers=1, num_attention_heads=2, eos_token_id=151643)
    cfg.update({'rope_parameters': {'rope_theta': 1000000}} if nested_rope else {'rope_theta': 1000000})
    p.write(tmp_path/'config.json', cfg)
    save_file({'model.embed_tokens.weight': np.zeros((10, size), dtype='float32'),
               'model.layers.0.input_layernorm.weight': np.ones((size,), dtype='float32')}, tmp_path/'model.safetensors')
    assert data.checkpoint(tmp_path)['sha256']
    p.write(tmp_path/'model.safetensors.index.json', {'weight_map': {'bad': 'missing.safetensors'}})
    with pytest.raises(ValueError, match='Shard coverage'):
        data.checkpoint(tmp_path)


def test_incomplete_comparison_rejected(tmp_path):
    p.write(tmp_path/'status.json', {'stage': 'FAILED'})
    with pytest.raises(ValueError, match='not complete'):
        r.compare(tmp_path, tmp_path, tmp_path/'comparison')
