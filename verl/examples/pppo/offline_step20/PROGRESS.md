# Step20 pause and offline evaluation progress

Snapshot: 2026-11-13, after the real GPU diagnostic passed. This is an execution
receipt, not a claim that the five formal evaluation arms have completed.

- Training: **PAUSED_AT_STEP20**, Ray job `23000000`; driver and task runner exited.
- Checkpoint: `global_step_20`, committed; all 50 recovery files match sidecar
  membership/sizes. HF weights fully hashed (see `evidence/checkpoint_preflight.json`).
- Actual optimizer updates: **320**, including 16 in Step20. No Step21 update.
- Step20 loss: 0.0076893703; gradient norm: 0.140885924; holdout accuracy:
  **0.35546875**. Eta remains 15%.
- Both original nodes released all 16 GPUs, verified by Ray resources and
  node-local GPU memory/utilization and compute-process queries.
- CPU tests: 21 evaluation checks and 10 pause guards passed.
- GPU diagnostic: **PASS**, 256/256 responses, eight shards of 32 responses.
- Formal PPPO Step20 four-benchmark evaluation: **RUNNING** (19,200 responses).
- Four historical Step300 MATH500 arms: queued sequentially (64,000 responses).
- Training stays paused after evaluation. There is no automatic resumption.

The first diagnostic attempt failed before generating any responses because CUDA
was initialized before a fork. Logs and receipts are retained under
`failed_attempts/startup_fork_20261113T2000Z/`. Switching vLLM worker startup to
`spawn` fixed initialization; the subsequent full diagnostic passed. Sampling,
model precision, frozen prompts, request seeds and verifier did not change.

The Step20 process-lineage estimate is **641.94146 GPU-hours**, versus NCBR's
628.64973, approximately **+2.114%**. The Step20 sum of step timers is 636.97855
GPU-hours; these scopes differ and must not be interchanged. The trailing work
between committed save and driver termination is separately recorded as
0.12955 GPU-hours. No exact compute-equivalence claim is made.

Successful diagnostic reserved-head-GPU cost: 0.487923 GPU-hours. Failed startup
cost: 0.034031 GPU-hours, preserved separately in `failed_attempt_costs.json` and
additional to the successful diagnostic/formal arms in the primary cost report.
The idle worker node is not counted as an active evaluation worker. All costs
are distinct from training costs.

Live artifacts:
`outputs/pppo_ncbr_matched_16xa10040g_s300_ckpt1_v2/offline_eval_step20_h8192_math500_v1/`.
Use each arm's `status.json` and `scores.json` for completion; the launch receipt
or this snapshot alone does not establish completed inference. The finite
execution job sends authorized milestone, failure and completion notifications.
