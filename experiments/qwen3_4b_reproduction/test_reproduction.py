"""CPU contract tests. No Ray initialization, model load, or GPU allocation."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


@pytest.fixture
def settings(tmp_path):
    return {'MODEL_PATH': '/model with spaces/base', 'TRAIN_FILE': '/data/train.parquet',
            'VAL_FILE_1': '/data/a24.parquet', 'VAL_FILE_2': '/data/a25.parquet',
            'OUTPUT_ROOT': str(tmp_path), 'CACHE_ROOT': str(tmp_path / 'cache'),
            'GPUS_PER_NODE': 8, 'RAY_CPUS': 32, 'RAY_OBJECT_STORE_GB': 16,
            'RUN_TAG': 'test', 'PROJECT_NAME': 'test'}


@pytest.mark.parametrize('arm,context,interval', [('dapo', 3072, 100), ('dapo4k', 5120, 100), ('ncbr', 9216, 50)])
def test_historical_recipe_and_fresh_start(settings, arm, context, interval):
    c = run.build_config(settings, arm)
    run.validate_config(c)
    assert c.actor_rollout_ref.model.path == settings['MODEL_PATH']
    assert (c.data.train_batch_size, c.data.gen_batch_size, c.actor_rollout_ref.rollout.n) == (256, 768, 8)
    assert c.actor_rollout_ref.actor.ppo_mini_batch_size == 16
    assert c.actor_rollout_ref.rollout.max_model_len == context
    assert c.trainer.total_training_steps == 300
    assert c.trainer.save_freq == interval
    assert c.trainer.resume_mode == 'disable' and c.trainer.resume_from_path is None
    assert c.trainer.val_before_train and c.trainer.test_freq == 10
    assert c.ray_kwargs.ray_init.address == 'local'
    assert c.data.seed is None
    assert c.actor_rollout_ref.actor.fsdp_config.seed == 42
    assert c.actor_rollout_ref.rollout.boundary_return.scheduler == 'fixed_wave'
    assert c.reward.reward_kwargs.overlong_buffer_cfg.enable
    assert c.reward.reward_kwargs.overlong_buffer_cfg.len == (819 if arm == 'dapo4k' else 410)
    assert c.reward.reward_kwargs.max_resp_len == (4096 if arm == 'dapo4k' else 2048)
    assert c.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert not c.actor_rollout_ref.rollout.val_kwargs.do_sample
    assert c.actor_rollout_ref.actor.optim.lr == 1e-6
    # Only metadata additions/explicit allowlisted changes are accepted.
    run.historical_diff(c, arm)


@pytest.mark.parametrize('arm', ['dapo', 'dapo4k', 'ncbr'])
def test_smoke_retains_full_recipe(settings, arm):
    c = run.build_config(settings, arm, smoke=True)
    run.validate_config(c)
    assert c.trainer.total_training_steps == c.trainer.test_freq == c.trainer.save_freq == 2
    assert (c.data.train_batch_size, c.data.gen_batch_size) == (256, 768)
    assert 'smoke_s2' in c.trainer.default_local_dir


@pytest.mark.parametrize('gpus', [1, 3, 6, 24])
def test_invalid_topology(settings, gpus):
    settings['GPUS_PER_NODE'] = gpus
    with pytest.raises((ValueError, AssertionError)):
        run.validate_config(run.build_config(settings, 'dapo'))


@pytest.mark.parametrize('path,value', [('data.train_batch_size', 128),
                                      ('actor_rollout_ref.rollout.n', 16),
                                      ('reward.reward_kwargs.overlong_buffer_cfg.enable', False),
                                      ('actor_rollout_ref.rollout.max_model_len', 9216)])
def test_silent_scientific_drift_rejected(settings, path, value):
    from omegaconf import OmegaConf
    c = run.build_config(settings, 'dapo')
    OmegaConf.update(c, path, value)
    with pytest.raises(ValueError, match='Unexpected historical change'):
        run.historical_diff(c, 'dapo')


def test_missing_and_changed_assets(settings, tmp_path, monkeypatch):
    asset = tmp_path / 'config.json'
    asset.write_text('expected')
    manifest = {'model': {'config.json': {'bytes': 8, 'sha256': run.sha256(asset)}}, 'data': {}}
    monkeypatch.setattr(run, 'read_json', lambda _: manifest)
    settings['MODEL_PATH'] = str(tmp_path / 'missing')
    with pytest.raises(ValueError, match='Missing Base'):
        run.check_assets(settings)
    settings['MODEL_PATH'] = str(tmp_path)
    asset.write_text('modified')
    with pytest.raises(ValueError, match='hash mismatch'):
        run.check_assets(settings)


def test_duplicate_output_not_overwritten(settings):
    out = run.run_dir(settings, 'dapo')
    out.mkdir()
    marker = out / 'keep.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        run.launch(settings, 'dapo', False, {})
    assert marker.read_text() == 'preserve'


def test_missing_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError, match='Missing dataloader'):
        run.verify_checkpoint(tmp_path, 300, 8)


def test_incomplete_training_not_success(tmp_path):
    (tmp_path / 'metrics.jsonl').write_text(json.dumps({'step': 1, 'data': {'training/global_step': 1}}) + '\n')
    with pytest.raises(ValueError, match='Incomplete trainer steps'):
        run.verify_run(tmp_path, 'dapo', 300, 8, False)


def smoke_metrics(tmp_path, arm='dapo', loss=0.1):
    records = [{'step': 0, 'data': {'val-core/math_dapo/acc/mean@1': 0.2}}]
    for step in (1, 2):
        records.append({'step': step, 'data': {'training/global_step': step,
                        'actor/pg_loss': loss, 'actor/grad_norm': 1.0,
                        'boundary_return/continuation_request_count': 5 if arm == 'ncbr' else 0}})
    records.append({'step': 2, 'data': {'val-core/math_dapo/acc/mean@1': 0.2}})
    (tmp_path / 'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in records) + '\n')
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/latest_checkpointed_iteration.txt').write_text('2')


@pytest.mark.parametrize('arm', ['dapo', 'dapo4k', 'ncbr'])
def test_completion_requires_steps_validation_and_checkpoints(tmp_path, monkeypatch, arm):
    smoke_metrics(tmp_path, arm)
    calls = []
    def checkpoint(out, step, world_size):
        calls.append((step, world_size))
        return {'step': step}
    monkeypatch.setattr(run, 'verify_checkpoint', checkpoint)
    assert run.verify_run(tmp_path, arm, 2, 8, True)['state'] == 'complete'
    assert calls == [(2, 8)]


@pytest.mark.parametrize('loss', [None, float('nan'), float('inf'), [float('nan')]])
def test_invalid_loss_never_complete(tmp_path, loss):
    smoke_metrics(tmp_path, loss=loss)
    with pytest.raises(ValueError, match='[Nn]on-finite|health metric'):
        run.verify_run(tmp_path, 'dapo', 2, 8, True)


def test_ncbr_without_continuation_not_accepted(tmp_path):
    smoke_metrics(tmp_path)
    with pytest.raises(ValueError, match='continuation count'):
        run.verify_run(tmp_path, 'ncbr', 2, 8, True)


def test_busy_gpu_rejected(settings, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    def query(args, **kwargs):
        return '\n'.join(f'{i}, GPU-{i}, NVIDIA H100, 81920, 5000, 80' for i in range(8))
    monkeypatch.setattr(run.subprocess, 'check_output', query)
    with pytest.raises(ValueError, match='GPU busy'):
        run.check_gpu(settings)


def test_gpu_count_mismatch(settings, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: '0, GPU-0, NVIDIA H100, 81920, 0, 0')
    with pytest.raises(ValueError, match='count/duplicates'):
        run.check_gpu(settings)


def test_missing_dependency_reports_name(monkeypatch):
    monkeypatch.setattr(run.metadata, 'version', lambda _: (_ for _ in ()).throw(run.metadata.PackageNotFoundError()))
    with pytest.raises(ValueError, match='Missing dependency'):
        run.environment()


def test_fresh_venv_requires_portable_crypto_dependencies(monkeypatch):
    expected = run.read_json(run.ROOT / 'provenance/dependencies.json')['versions']
    def version(name):
        if name in expected:
            return expected[name]
        raise run.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(run.metadata, 'version', version)
    monkeypatch.setattr(run.sys, 'prefix', '/isolated-new-venv')
    with pytest.raises(ValueError, match='Missing portable dependency: cffi'):
        run.environment()


def test_reference_assets_have_historical_repeats():
    data = run.read_json(run.ROOT / 'provenance/assets.json')['data']
    assert [data[k]['rows'] for k in ('TRAIN_FILE', 'VAL_FILE_1', 'VAL_FILE_2')] == [1791700, 960, 30]


def test_source_and_recipe_integrity():
    run.verify_source()
    run.verify_recipe()


def test_dapo4k_lengths_schedule_and_checkpoint_state(settings):
    c = run.build_config(settings, 'dapo4k')
    assert c.trainer.nnodes == 1 and c.trainer.n_gpus_per_node == 8
    assert c.data.max_prompt_length == 1024
    assert c.data.max_response_length == c.actor_rollout_ref.rollout.response_length == 4096
    assert c.actor_rollout_ref.rollout.max_model_len == 5120
    assert c.actor_rollout_ref.rollout.boundary_return.mode == 'off'
    assert c.actor_rollout_ref.actor.optim.lr_scheduler_type == 'constant'
    assert c.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert c.actor_rollout_ref.actor.checkpoint.save_contents == ['model', 'optimizer', 'extra', 'hf_model']
    assert c.actor_rollout_ref.actor.checkpoint.load_contents == ['model', 'optimizer', 'extra']
    assert c.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert c.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 2
    assert c.actor_rollout_ref.actor.fsdp_config.optimizer_offload
    assert not c.actor_rollout_ref.actor.fsdp_config.param_offload
    assert 'dapo4k_s300_' in c.trainer.default_local_dir
    smoke = run.build_config(settings, 'dapo4k', smoke=True)
    assert smoke.data.max_response_length == 4096
    assert smoke.reward.reward_kwargs.overlong_buffer_cfg.len == 819


@pytest.mark.parametrize('path,value', [
    ('reward.reward_kwargs.overlong_buffer_cfg.len', 410),
    ('reward.reward_kwargs.max_resp_len', 2048),
    ('data.max_response_length', 2048),
    ('actor_rollout_ref.rollout.response_length', 2048),
    ('actor_rollout_ref.rollout.max_model_len', 3072)])
def test_dapo4k_rejects_2k_drift(settings, path, value):
    from omegaconf import OmegaConf
    c = run.build_config(settings, 'dapo4k')
    OmegaConf.update(c, path, value)
    with pytest.raises(ValueError, match='Unexpected historical change'):
        run.historical_diff(c, 'dapo4k')


def test_dapo4k_formal_checkpoint_interval(tmp_path, monkeypatch):
    records = [{'step': step, 'data': {'training/global_step': step, 'actor/pg_loss': .1,
                'actor/grad_norm': 1.}} for step in range(1, 301)]
    records += [{'step': step, 'data': {'val-core/math_dapo/acc/mean@1': .2}} for step in range(0, 301, 10)]
    (tmp_path / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/latest_checkpointed_iteration.txt').write_text('300')
    calls = []
    def verify(out, step, size):
        calls.append((step, size))
        return {'step': step}
    monkeypatch.setattr(run, 'verify_checkpoint', verify)
    assert run.verify_run(tmp_path, 'dapo4k', 300, 8, False)['state'] == 'complete'
    assert calls == [(100, 8), (200, 8), (300, 8)]


def test_all_remains_original_two_arms(settings, monkeypatch, capsys):
    monkeypatch.setattr(run, 'settings', lambda: settings)
    run.main(['config', 'all'])
    output = capsys.readouterr().out
    assert 'PASS config: dapo,' in output and 'PASS config: ncbr,' in output
    assert 'PASS config: dapo4k,' not in output


def test_dapo4k_disk_budget_is_three_checkpoints(settings, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    settings['OUTPUT_ROOT'] = str(tmp_path)
    settings['CACHE_ROOT'] = str(tmp_path)
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda path: SimpleNamespace(free=250_000_000_000))
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=512 * (1 << 30)))
    run.check_resources(settings, ['dapo4k'], False)
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda path: SimpleNamespace(free=249_999_999_999))
    with pytest.raises(ValueError, match='Insufficient disk'):
        run.check_resources(settings, ['dapo4k'], False)
