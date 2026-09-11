# Qwen3-1.7B eight-GPU NCBR validation

Production source is pinned to `37a812d41ec7c1c5a070ee7741bed819f60aed55`, oracle to
`d23da0e17830c296ae6e375793a7ccea98870bb3`, and disabled baseline to
`bfa08860fee9f4febaf7aa1041f0e7eb0ac09cd5`. All production files are unchanged.
This directory contains validation orchestration and process-local compatibility/audit code.
Run from a fresh isolated clone; do not reuse evidence filenames or active training resources.
The report and raw artifacts for this run are under `evidence/`.

The user changed the original two-GPU plan to all eight GPUs. Actor uses eight-rank FSDP;
rollout uses eight TP=1 replicas. Each launch waits for idle devices and performs fresh
UUID-bound CUDA probes. The UUID compatibility shim resolves NVML physical indices without
changing CUDA_VISIBLE_DEVICES. It loads lazily after Ray assigns devices. Shared packages
are not patched. Ray, IPC, cache and outputs stay inside this clone. There is no global
`ray stop`, checkpoint save, benchmark evaluation, or modification of shared models.

Preparation and primary commands:

```bash
python validation/prepare.py  # fresh run only; this run reused and revalidated the frozen manifest
python validation/gate.py
python verl/tests/experimental/ncbr_portable/verify.py --output-dir <fresh-private-output>
python validation/launch.py --h 128 --l 512 --count 8
python validation/replay.py --h 128 --l 512 --suffix v2
python validation/replay.py --h 128 --l 512 --suffix v3
python validation/launch.py --h 2048 --l 8192 --count 32
python validation/replay.py --h 2048 --l 8192 --suffix v2
python validation/replay.py --h 2048 --l 8192 --suffix v3
python validation/launch_actor.py --h 128
python validation/launch_actor.py --h 2048
python validation/check_actor_modes.py
python validation/tail_isolation.py  # bind an idle permitted GPU by UUID
python validation/launch_trainer.py --entry dapo --mode off
python validation/launch_trainer.py --entry dapo --mode shadow
python validation/remaining_cases.py
python validation/check_reference.py
python validation/live_suite.py
python validation/fault_suite.py
python validation/analyze_cases.py
python validation/check_receipts.py
python validation/report.py
python validation/cleanup_owned.py  # inspect after all suites exit
python validation/finalize.py  # requires no active owned processes
```

`nccl_probe.py` is launched with torchrun --standalone --nnodes=1 --nproc-per-node=8,
with CUDA_VISIBLE_DEVICES set to the eight frozen UUIDs. Each rank must report its
expected UUID and an all-reduce result of 36. Commands and resolved trainer YAMLs
are retained with each case. Do not run the commands concurrently on the same GPUs.

The ordinary trainer cases use H2048/L8192, n=4, base seed42, temperature1/top_p1/top_k-1,
ignore_eos=false, max_model_len9216, rollout memory0.35, eager execution, continuation
concurrency4/request batch8/timeout600, reward workers2/long chunks3, buffer410,
AdamW lr1e-6, mini-batch2 prompts and microbatch1 per GPU. Each vLLM engine is seeded42;
continuation seeds use the unchanged stable derivation. Trainer UUIDs are deterministic
UUID5 counters; the frozen capture uses prompt SHA256 UIDs. Never alter sampling or
supplement prompts based on coverage. Independent generations are not presumed equal.

The direct capture is preparatory. Frozen replay compares source runtime/adapter with hook,
including real verifier outputs, exact tensors, request identity, masks and retained order.
Real FSDP repeats verify registered vanilla/GSPO losses and gradients. GPU tail isolation
uses a separately labeled synthetic task-score increment, not fabricated verifier output.

Live oracle diagnostics call the original runtime and hook against the same real H batch,
weights and service, compare request/mask identity, and freeze original service output for
an exact adapter comparison. Smoke uses the first8 prompts; full uses the first32. They
stop deliberately after comparison and before any optimizer update. Exit code1 plus the
live-validation-complete receipt is expected, and is checked by live_suite.py.

Fault cases use the first2 fixed prompts at H128/L512 and real service/callback boundaries.
Faults are explicitly recorded as controlled injections. fault_suite.py requires no AdamW
step or new version after an injected fault, and no sleep after unconfirmed release.
Expected fault exits are nonzero. No-cap cases remain uncovered.

The audit wraps existing calls to save raw batches, verifier outputs, actual gradients,
weight checksums, publication and cleanup receipts. It does not change loss, advantage,
detector or reward mathematics. The live oracle's extra service calls are diagnostic-only.
Early runner failures and their original logs are retained; see the report for limits and
coverage. Cleanup script defaults to inspection and targets only this run's audit marker.

For a stopped fault suite, preserve the failed case directory and original logs before
`python validation/fault_suite.py --resume`; completed PASS cases are not rerun.
The launcher requires two idle snapshots and no compute processes before its CUDA gate.
