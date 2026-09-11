"""Bounded continuation dispatch with release acknowledgement as the slot boundary."""
import asyncio
from collections import deque


async def schedule(requests, concurrency, batch_size, start, generate, release, cleanup):
    """Own lifecycle transitions; callbacks own remote operations and error attestation.

    Dispatch and release tasks are never cancelled: their handles/acknowledgements
    must be collected before remote cleanup. Generation is settled by cleanup only
    after remote abort/drain. Results retain input order regardless of completion.
    """
    source = iter(enumerate(requests))
    queued = deque()
    active = {}
    all_tasks = []
    results = [None] * len(requests)
    exhausted = False

    def launch(awaitable, phase, index, handle=None):
        task = asyncio.create_task(awaitable)
        active[task] = (phase, index, handle)
        all_tasks.append(task)

    def fill():
        nonlocal exhausted
        while len(active) < concurrency:
            if not queued and not exhausted:
                for _ in range(batch_size):
                    item = next(source, None)
                    if item is None:
                        exhausted = True
                        break
                    queued.append(item)
            if not queued:
                break
            index, request = queued.popleft()
            launch(start(request), 'start', index)

    try:
        fill()
        while active:
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            # Observe every ready failure before making any new remote call.
            for task in done:
                task.result()
            for task in sorted(done, key=lambda task: active[task][1]):
                phase, index, handle = active.pop(task)
                value = task.result()
                if phase == 'start':
                    launch(generate(*value), 'generate', index, value)
                elif phase == 'generate':
                    results[index] = value
                    launch(release(*handle), 'release', index, handle)
            fill()
        return results
    except BaseException as error:
        async def settle():
            settled = await asyncio.gather(
                *(task for task, (phase, _, _) in active.items() if phase in ('start', 'release')),
                return_exceptions=True,
            )
            for secondary in settled:
                if isinstance(secondary, BaseException) and secondary is not error:
                    error.add_note(f"Concurrent lifecycle failure: {type(secondary).__name__}: {secondary}")
                    if not getattr(secondary, "boundary_remote_cleanup_attested", True):
                        error.boundary_remote_cleanup_attested = False
            await cleanup(error, all_tasks)
        # Repeated caller cancellation cannot interrupt ownership transfer/cleanup.
        settling = asyncio.create_task(settle())
        while not settling.done():
            try:
                await asyncio.shield(settling)
            except asyncio.CancelledError:
                continue
        settling.result()
        raise
