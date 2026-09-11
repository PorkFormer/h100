# Continuation scheduler validation — in progress

Implementation is isolated at `/tmp/ncbr_scheduler_20260911`, based on `c557509`.
The active shared checkout is unchanged. Default remains `fixed_wave`.

- Event-controlled scheduler and runtime fault tests pass, including slow dispatch,
  bounded slots, release acknowledgement, cross-batch refill, identity ordering,
  missing/duplicate/version failures, timeout, abort/drain/release order, secondary
  cleanup failures and repeated cancellation. Tests were introduced before code;
  initial collection failure is preserved in `evidence/tests_before.log`.
- Original 190 CPU tests and both unchanged frozen characterizations passed.
  Complete count: 215 passed, plus 4 independent runner gate/ownership tests.
- Frozen 2048-row pool (78 continuations) and real natural-correction batch
  (17 continuations) pass exact old-4/new-4/new-8 replay for off/shadow/replace.
  Requests/calls, token/masks, raw/effective rewards, filter order, all actor tensors
  and GRPO advantages are equal. The actual retained 8-row actor batch is bound
  separately; no replacement data or synthetic verifier outputs were added.
- Fresh eight-device CUDA computation and eight-rank NCCL all-reduce pass.
  An earlier CUDA gate refused transient 4 MiB allocations on all devices; its
  evidence remains preserved and the gate was repeated only after all became idle.
- GPU actor scheduler comparison: PASS. New concurrency 4 and 8 each match all
  96 old-scheduler vanilla/GSPO loss-input and gradient-shard records exactly.
  All three variants pass their own eight-rank repeated-gradient checks.
  The real 0-to-1 event changes 4 advantage rows and all 8 gradient shards for
  both vanilla and GSPO; shadow remains exactly equal to off.
- Six paired scheduler and six paired capacity comparisons: RUNNING/PENDING.
- Six four-update trainer cases: PENDING; no integration PASS claimed yet.

The first old repeat was launched before the runner moved load-balancer readiness
into startup measurement. Its tokens remain valid repeat evidence; its timing is
not part of either six-pair performance comparison. All paired timing instances
explicitly await service and load-balancer readiness before identical warmup.

Raw results and immutable input hashes are retained under `evidence/` and
`manifest.json`. No performance or complete-integration claim is made by this
interim report. See the final report revision and per-case receipts after completion.
