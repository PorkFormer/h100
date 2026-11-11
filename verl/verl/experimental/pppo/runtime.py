"""All-prefix, independent n=1 requests with bounded lifecycle ownership."""
import asyncio
import hashlib
import json
from dataclasses import dataclass
from verl.experimental.natural_continuation_boundary_return.scheduler import schedule


@dataclass(frozen=True)
class Request:
    request_id: str
    parent: int
    branch: int
    occurrence: str
    policy_version: int
    prompt: tuple
    prefix: tuple
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class Generation:
    request: Request
    tail: tuple
    stop_reason: str
    finish_reason: str


def build_requests(batch, g, cap, version, seed, max_model_len):
    requests = []
    prompt_width = batch.batch["prompts"].shape[-1]
    for parent in range(len(batch)):
        prompt_mask = batch.batch["attention_mask"][parent, :prompt_width].bool()
        prompt = tuple(batch.batch["prompts"][parent][prompt_mask].tolist())
        h = int(batch.batch["pppo_prefix_length"][parent])
        prefix = tuple(batch.batch["responses"][parent, :h].tolist())
        occurrence = str(batch.non_tensor_batch["trajectory_id"][parent])
        if not prompt or len(prompt) + cap > max_model_len or h >= cap:
            raise ValueError("invalid continuation context")
        for branch in range(g):
            digest = hashlib.sha256(json.dumps(["pppo", seed, version, occurrence, branch]).encode()).hexdigest()
            requests.append(Request(f"pppo-{digest}", parent, branch, occurrence, version,
                                    prompt, prefix, cap - h, int(digest[:16], 16) % (2**31)))
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("duplicate PPPO request identity")
    return requests


def validate_result(request, outputs):
    if len(outputs) != 1:
        raise ValueError("independent n=1 request returned wrong branch count")
    output = outputs[0]
    extra = output.extra_fields or {}
    if int(extra.get("branch_id", 0)) != 0 or extra.get("global_steps") != request.policy_version:
        raise ValueError("PPPO branch or policy version mismatch")
    stop = str(getattr(output.stop_reason, "value", output.stop_reason)).lower()
    finish = str(extra.get("finish_reason", "")).lower()
    if stop in ("none", "") or not finish or any(x in stop + finish for x in ("abort", "error", "timeout", "cancel")):
        raise ValueError("invalid continuation terminal state")
    tail = tuple(int(x) for x in output.token_ids)
    if len(tail) > request.max_tokens:
        raise ValueError("overlong continuation")
    if not tail and not any(x in (stop, finish) for x in ("eos", "stop", "completed")):
        raise ValueError("invalid empty tail")
    return Generation(request, tail, stop, finish)


async def generate_chunk(requests, client, sampling_params, concurrency=128, timeout=600):
    handles, result_tasks = [], []
    completed, released = set(), set()

    async def start(request):
        params = dict(sampling_params, n=1, seed=request.seed, max_tokens=request.max_tokens)
        handle = await client.start_grouped(request.request_id, prompt_ids=list(request.prompt + request.prefix),
                                            sampling_params=params, routing_key=f"pppo-{request.occurrence}")
        handles.append(handle)
        return request, handle

    async def generate(request, handle):
        task = asyncio.create_task(handle.result())
        result_tasks.append(task)
        outputs = await asyncio.wait_for(asyncio.shield(task), timeout)
        completed.add(id(handle))
        return validate_result(request, outputs)

    async def release(request, handle):
        await handle.release()
        released.add(id(handle))

    async def cleanup(error, tasks):
        failures = []
        active = [h for h in handles if id(h) not in completed]
        for operations in ([h.abort() for h in active],
                           [h.drain() for h in {str(h.server_id): h for h in handles}.values()]):
            results = await asyncio.gather(*operations, return_exceptions=True)
            failures.extend(x for x in results if isinstance(x, BaseException))
        if not failures:
            results = await asyncio.gather(*(h.release() for h in handles if id(h) not in released),
                                           return_exceptions=True)
            failures.extend(x for x in results if isinstance(x, BaseException))
        for task in [*tasks, *result_tasks]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, *result_tasks, return_exceptions=True)
        error.pppo_cleanup_attested = not failures and getattr(error, "boundary_remote_cleanup_attested", True)
        for failure in failures:
            error.add_note(f"PPPO cleanup failure: {failure!r}")
        print(f"PPPO cleanup_attested={error.pppo_cleanup_attested}", flush=True)

    results = await schedule(requests, concurrency, len(requests), start, generate, release, cleanup)
    # release ACKs above are necessary; drain also attests no outstanding backend ownership.
    await asyncio.gather(*(h.drain() for h in {str(h.server_id): h for h in handles}.values()))
    if {x.request.request_id for x in results} != {r.request_id for r in requests}:
        raise ValueError("incomplete PPPO result set")
    return results
