# Continuation scheduler validation

Base: `c557509845f5aacbde9b990ba9157b66502a294b`. Production defaults remain
`boundary_return.scheduler=fixed_wave`. Explicit candidate configurations:

```yaml
actor_rollout_ref:
  rollout:
    boundary_return:
      scheduler: work_conserving
      max_concurrent_requests: 4  # evaluate 8 separately
      request_batch_size: 8
      request_timeout_seconds: 600
```

Use `ncbr.enable=true` with an explicit `shadow` or `replace` mode. A disabled
hook is inert. No detector, request content, route selection algorithm, sampling,
verifier, reward, actor prefix, optimizer or production determinism setting changes.

Run from the isolated repository root:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 python verl/tests/experimental/ncbr_portable/verify.py --output-dir scheduler_validation/evidence/cpu_fresh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scheduler_validation/replay.py
python scheduler_validation/freeze.py
python scheduler_validation/launch.py --case nccl_initial --kind nccl
python scheduler_validation/launch.py --case repeat_old_0 --scheduler fixed_wave --concurrency 4
python scheduler_validation/performance_report.py
```

Every output name is exclusive. Preserve failed runs and use a new explicit name
for retries. The manifest binds original saved data/model/dependencies; runners
read prior artifacts and write only inside this independent checkout or a unique
`/tmp/ncs_<case>` runtime directory. They never reset GPUs or stop shared Ray.
Eight devices must be idle and pass fresh CUDA probes before every GPU launch.

Performance protocol: two old repeats, then six pairs comparing fixed-wave 4 with
work-conserving 4, then six distinct pairs comparing work-conserving 4 with 8.
Alternate A/B launch order by pair. Each instance starts eight fresh services and
uses identical warmup. Keep the original sticky least-inflight load balancer and
save actual routing. The paired statistic is the arithmetic mean of per-pair
speed ratios; 10,000 paired bootstrap resamples use seed 42. Claim performance
only for exact generated work, ratio >= 1.05 and 95% CI lower bound > 1. Otherwise
all timing ratios are diagnostic only. Capacity effects are a separate comparison.

`evidence/` holds raw output, first failures, generated tokens, event traces,
CUDA/NCCL receipts, frozen comparisons and per-run memory samples. Historical
`validation/RESULTS.md` belongs to the base run; it is not this scheduler's result.
