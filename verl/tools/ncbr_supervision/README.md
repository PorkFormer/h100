# NCBR resolved-config startup and supervision

These utilities preserve an already migrated training recipe and supervise one explicitly authorized attempt. They accompany commit `5789e69`, which adds baseline-compatible naive GRPO/GSPO validation and bounded training audit hooks. They do not launch a sweep or retry a failed experiment automatically.

## Resolved configuration entry

```bash
python verl/tools/ncbr_supervision/resolved_entry.py /path/to/resolved.yaml --validate-only
python verl/tools/ncbr_supervision/resolved_entry.py /path/to/resolved.yaml
```

The entry validates the modern configuration, checks that validation leaves it unchanged, and calls `run_ppo` directly. Repeating Hydra's legacy migration on an `actual_resolved_config.yaml` previously failed with `Missing key reward_model`. Legacy configurations must still use the normal training entry.

Both inherited and Ray runtime `VLLM_PORT` are rejected: a shared fixed port caused TP2 replicas to collide with `EADDRINUSE`. Leave this variable unset to retain the baseline per-replica dynamic allocator. No sampler, optimizer, reward, loss, or seed setting is changed.

## Local supervision

`supervision.py` is the exact implementation deployed in the isolated r3 controller. Import `pulse` in the supervisor and `watchdog` in a separate local process. The parent must launch the training driver with `start_new_session=True`, pass its process-group ID, refresh `pulse(stage_heartbeat)` every five seconds, and monitor the watchdog process itself. Stop the watchdog via its `done` marker when the driver exits; continue checkpoint and resource-release acceptance separately.

Use a unique local directory under `/tmp` for each attempt's heartbeat and done marker. `NCBR_PIPELINE_LEASE`, when set, must point to another locally refreshed JSON heartbeat produced by `pulse`. Both writers and readers must run on the same machine and boot; monotonic clocks are not cross-node timestamps. Optional `pulse(path, mirror)` copies are for shared audit only. Do not reuse done markers or paths across attempts.

The watchdog checks parent liveness and monotonic heartbeat age, preserving the 90-second timeout. Temporary read errors accumulate continuous unavailability instead of immediately masquerading as parent death. Confirmed parent exit and expired leases still stop the owned process group. A local stop receipt is written before SIGTERM/SIGKILL; it records the precise reason, errno, parent identity and heartbeat ages, and is then mirrored to the run directory. It never stops Ray or unrelated process groups.

The r2 incident is confirmed as a watchdog-issued SIGTERM, but its old generic receipt cannot identify which probe triggered it. NFS visibility or read failure remains a hypothesis, not an established root cause. Regression tests cover transient I/O, both lease expirations, parent exit, and receipt-before-signal ordering. This is heartbeat/process supervision, not a replacement for the training-level OOM, policy-version, numerical, 60-minute no-progress, checkpoint and resource gates.

## Verification

```bash
PYTHONPATH=verl/tools/ncbr_supervision python -m pytest -q verl/tools/ncbr_supervision
FLASHINFER_WORKSPACE_BASE=/tmp/ncbr-check/flashinfer PYTHONPYCACHEPREFIX=/tmp/ncbr-check/pycache \
  python verl/tests/experimental/ncbr_portable/verify.py --output-dir /tmp/ncbr-portable-check
```

Experiment recipes, credentials, live manifests, logs, checkpoint weights, GPU snapshots and machine-specific dispatch scripts are intentionally kept in the local experiment audit directories. Running jobs remain bound to their original source and file hashes; this publication does not update live code.

Validated on 2026-09-15: 12 supervision/entry tests passed; the portable suite passed 218 tests with both frozen golden files byte-identical. The actual GRPO and GSPO resolved formal configurations passed the portable entry in validate-only mode without initializing Ray. FlashInfer cache writes were redirected to `/tmp`.
