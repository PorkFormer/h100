"""Event-controlled lifecycle tests; no timing-based speed assertions."""
import asyncio
import pytest
from verl.experimental.natural_continuation_boundary_return.scheduler import schedule


def test_refill_across_batches_waits_for_release_and_preserves_order():
    async def check():
        ready = [asyncio.Event() for _ in range(6)]
        release = [asyncio.Event() for _ in range(6)]
        started = [asyncio.Event() for _ in range(6)]
        releasing = [asyncio.Event() for _ in range(6)]
        active = set()
        async def start(i):
            active.add(i)
            assert len(active) <= 2
            started[i].set()
            return (i,)
        async def generate(i):
            await ready[i].wait()
            return [i]
        async def free(i):
            releasing[i].set()
            await release[i].wait()
            active.remove(i)
        async def cleanup(error, tasks):
            pytest.fail(str(error))
        job = asyncio.create_task(schedule(range(6), 2, 2, start, generate, free, cleanup))
        await started[1].wait()
        for i in range(1, 6):
            ready[i].set()
            await releasing[i].wait()
            assert not started[i + 1].is_set() if i < 5 else True
            release[i].set()
            if i < 5:
                await started[i + 1].wait()
        ready[0].set(); release[0].set()
        assert await job == [[i] for i in range(6)]
        assert not active
    asyncio.run(asyncio.wait_for(check(), 3))


@pytest.mark.parametrize('failure', ['start', 'generate', 'release', 'cancel'])
def test_failure_settles_slow_dispatch_before_cleanup_and_never_refills(failure):
    async def check():
        slow = asyncio.Event(); entered = asyncio.Event(); failed = asyncio.Event()
        starts = []; handles = []; frees = []; cleaned = []
        async def start(i):
            starts.append(i)
            if i == 1:
                entered.set(); await slow.wait()
            if i == 0 and failure == 'start':
                await entered.wait(); failed.set(); raise ValueError('first failure')
            handles.append(i)
            return (i,)
        async def generate(i):
            await entered.wait()
            if i == 0 and failure == 'generate':
                failed.set(); raise ValueError('first failure')
            if failure == 'cancel':
                await asyncio.Event().wait()
            return [i]
        async def free(i):
            frees.append(i)
            if i == 0 and failure == 'release':
                failed.set(); raise ValueError('first failure')
        async def cleanup(error, tasks):
            assert 1 in handles
            cleaned.append(error)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        job = asyncio.create_task(schedule(range(5), 2, 2, start, generate, free, cleanup))
        await entered.wait()
        if failure == 'cancel':
            job.cancel()
        else:
            await failed.wait()
        # Permit coordinator to observe failure while second dispatch is still blocked.
        for _ in range(10):
            await asyncio.sleep(0)
        assert not cleaned
        slow.set()
        with pytest.raises(asyncio.CancelledError if failure == 'cancel' else ValueError):
            await job
        assert starts == [0, 1]
        assert len(frees) == len(set(frees))
        assert len(cleaned) == 1
    asyncio.run(asyncio.wait_for(check(), 3))


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'version', 'generation', 'timeout', 'release', 'drain', None])
@pytest.mark.parametrize('concurrency', [4, 8])
def test_runtime_faults_stop_dispatch_and_attest_remote_cleanup(fault, concurrency):
    import numpy as np
    from types import SimpleNamespace
    from test_config_runtime_on_cpu import _batch, _active_config, _normal_sampling
    from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations
    batch = _batch().repeat(12, interleave=True)
    batch.non_tensor_batch['trajectory_id'] = np.asarray([f't{i}' for i in range(len(batch))], dtype=object)
    events = []; handles = []
    class Handle:
        server_id = 'server'
        def __init__(self, rid):
            self.backend_request_id = rid
            self.index = len(handles)
            self.aborted = asyncio.Event()
            handles.append(self)
        async def result(self):
            if self.index == 0:
                if fault in ('generation', 'drain'):
                    raise ValueError('generation failed')
                if fault == 'timeout':
                    await self.aborted.wait()
                if fault == 'missing':
                    return []
            output = SimpleNamespace(token_ids=[1], stop_reason='stop', extra_fields={
                'branch_id': 0, 'global_steps': 8 if fault == 'version' and self.index == 0 else 7,
                'finish_reason': 'stop'})
            return [output, output] if fault == 'duplicate' and self.index == 0 else [output]
        async def abort(self):
            events.append(('abort', self.index)); self.aborted.set()
        async def drain(self):
            events.append(('drain', self.index))
            if fault == 'drain':
                raise RuntimeError('drain failed')
        async def release(self):
            events.append(('release', self.index))
            if fault == 'release' and self.index == 0:
                raise RuntimeError('release failed')
    class Client:
        async def start_grouped(self, rid, **kwargs):
            events.append(('start', len(handles)))
            return Handle(rid)
    cfg = _active_config(scheduler='work_conserving', max_concurrent_requests=concurrency,
                         request_batch_size=8, request_timeout_seconds=0.01)
    kwargs = dict(config=cfg, rollout_batch=batch, client=Client(), eos_token_id=2,
                  short_response_length=4, max_model_len=9, policy_version=7,
                  sampling_params=_normal_sampling())
    if fault is None:
        result = run_boundary_continuations(**kwargs)
        assert len(result.generations) == 12
        assert [g.request_id for g in result.generations] == [r.request_id for r in result.requests]
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)) as caught:
            run_boundary_continuations(**kwargs)
        assert caught.value.boundary_remote_cleanup_attested == (fault not in ('release', 'drain'))
        # Timeout permits other slots to finish before it is observed.
        if fault != 'timeout':
            assert len(handles) == concurrency
        drain = next(i for i, e in enumerate(events) if e[0] == 'drain')
        assert all(i < drain for i, e in enumerate(events) if e[0] == 'abort')
        if fault == 'drain':
            assert not any(e[0] == 'release' for e in events)
    released = [i for event, i in events if event == 'release']
    assert len(released) == len(set(released))


def test_scheduler_config_default_validation_and_disabled_identity():
    from verl.workers.config.rollout import BoundaryReturnConfig
    from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations
    assert BoundaryReturnConfig().scheduler == 'fixed_wave'
    with pytest.raises(ValueError, match='scheduler'):
        BoundaryReturnConfig(mode='shadow', scheduler='unknown').validate()
    assert run_boundary_continuations(config=BoundaryReturnConfig(mode='off', scheduler='unknown'),
                                     rollout_batch=None, client=None, eos_token_id=None,
                                     short_response_length=0, max_model_len=0,
                                     policy_version=0, sampling_params=None) is None


def test_secondary_dispatch_failure_attestation_survives_repeated_cancellation():
    async def check():
        slow = asyncio.Event(); entered = asyncio.Event(); failing = asyncio.Event(); cleanup_started = asyncio.Event()
        primary = ValueError('generation failed'); secondary = RuntimeError('lease release failed')
        secondary.boundary_remote_cleanup_attested = False
        async def start(i):
            if i == 1:
                entered.set(); await slow.wait(); raise secondary
            return (i,)
        async def generate(i):
            await entered.wait(); failing.set(); raise primary
        async def release(i):
            pytest.fail('must not release before cleanup')
        async def cleanup(error, tasks):
            cleanup_started.set()
            assert error is primary
            assert not error.boundary_remote_cleanup_attested
            await asyncio.gather(*tasks, return_exceptions=True)
        job = asyncio.create_task(schedule(range(5), 2, 2, start, generate, release, cleanup))
        await failing.wait()
        for _ in range(10): await asyncio.sleep(0)
        job.cancel(); await asyncio.sleep(0); job.cancel(); await asyncio.sleep(0)
        assert not cleanup_started.is_set()
        slow.set()
        with pytest.raises(ValueError) as caught: await job
        assert caught.value is primary
    asyncio.run(asyncio.wait_for(check(), 3))


@pytest.mark.parametrize('name', [
    'test_remote_abort_drain_release_happens_before_local_settle_and_preserves_primary_error',
    'test_timeout_shields_backend_result_until_explicit_remote_abort',
])
def test_existing_remote_cleanup_contract_with_new_scheduler(monkeypatch, capsys, name):
    import dataclasses
    import test_config_runtime_on_cpu as original
    config = original._active_config
    monkeypatch.setattr(original, '_active_config',
                        lambda **kwargs: dataclasses.replace(config(**kwargs), scheduler='work_conserving'))
    getattr(original, name)(capsys)


def test_release_failure_receipt_includes_the_failed_slot_owner(capsys):
    test_runtime_faults_stop_dispatch_and_attest_remote_cleanup('release', 4)
    audit = capsys.readouterr().out
    assert 'event=release_ack count=4 errors=1' in audit
    assert 'cleanup_attested=False' in audit
