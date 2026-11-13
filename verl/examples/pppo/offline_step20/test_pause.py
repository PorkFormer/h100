"""Exercise stop guards with synthetic files; never signal a real process."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location('pause_step20', Path(__file__).with_name('pause.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


@pytest.fixture
def run(tmp_path, monkeypatch):
    signals = []
    monkeypatch.setattr(p.os, 'kill', lambda *args: signals.append(args))
    monkeypatch.setattr(p, 'RUN', tmp_path)
    cp = tmp_path / 'checkpoints/global_step_20'
    hf = cp / 'actor/huggingface'
    hf.mkdir(parents=True)
    (hf / 'model.safetensors').write_bytes(b'fixture only')
    monkeypatch.setattr(p, 'ARMS', {'pppo_s20': (20, hf)})
    files = {'data.pt', 'pppo_driver_rng.pt'} | {
        f'actor/{kind}_world_size_16_rank_{rank}.pt'
        for kind in ('model', 'optim', 'extra_state') for rank in range(16)}
    for name in files:
        (cp / name).write_bytes(b'fixture only')
    p.write(cp / 'pppo_state.json', dict(status='committed', step=20, policy_version=20,
            files={name: 12 for name in files}))
    p.write(tmp_path / 'training_status.json', dict(last_successful_step=20,
            driver_pid=999999999, job_id='test', started_at=1))
    p.write(tmp_path / 'pipeline_status.json', dict(state='DISABLED_FOR_STEP20_PAUSE'))
    metrics = {'actor/pppo_optimizer_updates_total': 320,
               'actor/pppo_optimizer_updates_this_outer': 16, 'pppo/actor_rows': 2048,
               'train/retained_prompt_groups': 256, 'actor/grad_norm': 0.1,
               'actor/loss': 0.01, 'pppo/holdout_accuracy': 0.25}
    row = dict(step=20, metrics=metrics, timing_seconds={'step': 7200})
    (tmp_path / 'training_metrics.jsonl').write_text(json.dumps(row) + '\n')
    return tmp_path, cp, row, signals


@pytest.mark.parametrize('failure', ['early', 'updates', 'nan', 'holdout', 'uncommitted',
                                      'policy_version', 'missing_shard', 'coordinator', 'identity'])
def test_invalid_pause_never_signals(run, failure):
    root, cp, row, signals = run
    if failure == 'early':
        status = p.read(root / 'training_status.json'); status['last_successful_step'] = 19
        p.write(root / 'training_status.json', status)
    elif failure in ('updates', 'nan', 'holdout'):
        if failure == 'updates': row['metrics']['actor/pppo_optimizer_updates_total'] = 319
        elif failure == 'nan': row['metrics']['actor/loss'] = float('nan')
        else: del row['metrics']['pppo/holdout_accuracy']
        (root / 'training_metrics.jsonl').write_text(json.dumps(row) + '\n')
    elif failure in ('uncommitted', 'policy_version'):
        sidecar = p.read(cp / 'pppo_state.json')
        sidecar['status' if failure == 'uncommitted' else 'policy_version'] = 'bad'
        p.write(cp / 'pppo_state.json', sidecar)
    elif failure == 'missing_shard': (cp / 'actor/optim_world_size_16_rank_15.pt').unlink()
    elif failure == 'coordinator': p.write(root / 'pipeline_status.json', {'state': 'RUNNING'})
    with pytest.raises((RuntimeError, AssertionError, FileNotFoundError)):
        p.main()
    assert signals == []
    assert not (root / 'pause_step20_receipt.json').exists()


def test_valid_pause_records_only_target_signal(run, monkeypatch):
    root, cp, row, signals = run
    old_exists, old_bytes, old_text = Path.exists, Path.read_bytes, Path.read_text
    monkeypatch.setattr(Path, 'exists', lambda self: True if str(self).startswith('/proc/999999999/') else old_exists(self))
    monkeypatch.setattr(Path, 'read_bytes', lambda self: b'/repo/experiments/pppo_ncbr_matched/launch.py\x00formal' if str(self).startswith('/proc/999999999/') else old_bytes(self))
    monkeypatch.setattr(Path, 'read_text', lambda self, *a, **kw: '999999999 (test) Z' if str(self).startswith('/proc/999999999/') else old_text(self, *a, **kw))
    p.main()
    assert signals == [(999999999, p.signal.SIGTERM)]
    assert p.read(root / 'training_status.json')['state'] == 'PAUSED_AT_STEP20'
    receipt = p.read(root / 'pause_step20_receipt.json')
    assert receipt['checkpoint'] == str(cp)
    assert receipt['step20_metrics'] == row
