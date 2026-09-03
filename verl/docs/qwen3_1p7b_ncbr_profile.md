# Qwen3-1.7B Baseline / NCBR-v1 profiling protocol

This protocol is pinned to `Qwen/Qwen3-1.7B-Base` revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, H=2048, L=8192,
B256/G768/M16/N8 and seed 42. A stage manifest must bind the clean code SHA,
diagnostics mode, node identity, every node-local model file, Hugging Face
revision metadata, train/AIME2024/AIME2025 data hashes, and any frozen workload
or mechanism-panel artifact before a GPU process starts. AIME2024 and AIME2025
remain separate in every report.

The launcher is `examples/natural_continuation_boundary_return/run_qwen3_1p7b_profile_fsdp.sh`.
It accepts only the two arms, P0/P1/P2, and `calibration`, `gate0`, `profile`,
`acceptance`, or `formal_s300`. Calibration itself is fixed to Baseline/P0.
`formal_s300` additionally fails closed unless the later explicit authorization
token is present. Profiling and five-step acceptance never write inside the
formal S300 namespaces. The effective vLLM engine seed and NCBR seed are both
42; disabled interventions, including distillation and rollout correction, are
explicit in the resolved command.

The launcher requires `NCBR_NODE`, `NCBR_ARM`, `NCBR_PROFILE_CANDIDATE`,
`NCBR_STAGE`, `NCBR_STAGE_MANIFEST`, and `NCBR_DIAGNOSTICS_MODE`. A profile also
requires a fresh shared `NCBR_PROFILE_COORDINATION_DIR`. The one-shot controller
additionally requires the already compared cross-node minimum
`NCBR_OBJECT_STORE_BYTES`, common `NCBR_RAY_CPUS`, and a stage-specific
`NCBR_RECEIPT_DIR`.

## Stage boundaries

- Gate 0 runs three complete candidate-accumulation cycles and stops before
  old/ref log probabilities, advantage calculation, and actor update. Each
  cycle must retain exactly 256 UID groups with eight trajectories within ten
  normally completed candidate batches. Generation, reward, or Ray errors are
  system failures; only normal exhaustion is a Dynamic Sampling failure.
- Profiling starts at Base and each process is bounded to at most six steps.
  Step 1 is warmup. At Step 4 each live arm atomically publishes its Step 2-4
  wall CV in the fresh coordination directory and waits for its peer. If either
  arm exceeds 10%, both existing processes continue to Step 6 and Step 2-6 is
  used without deleting outliers; otherwise both stop at Step 4. A stale file,
  missing peer, identity mismatch, or timeout fails closed.
- Acceptance starts at Base, validates Step 0 and Step 5, saves Step 5, and
  verifies a complete loadable checkpoint. It then stops. It does not authorize
  S300, recovery, Step 600, or another seed.

## Diagnostics and profiling semantics

Actor diagnostics read current/old log probability, advantage, response mask,
old-policy categorical entropy, and retained-row labels already present in the
loss path. Micro-batches only add detached sufficient statistics. Each optimizer
step packs them into one vector and performs no more than one diagnostic DP
all-reduce. Reported fields include raw and numerically clamped log ratio,
numerical-clamp rate, PPO-bound exceedance, effective-ratio ESS, response thirds,
advantage sign, and retained NCBR cohorts. Eight 256-token entropy buckets carry
token-weighted and sequence-balanced entropy plus token/trajectory counts. No
extra forward is introduced.

NCBR replace rows carry `boundary_hit_cap`, `boundary_eligible`,
`boundary_applied`, `boundary_changed`, `boundary_recovered`,
`boundary_regressed`, `boundary_task_delta`, `boundary_group_unlocked`, UID and
trajectory ID through the actor batch. `boundary_group_newly_locked` is only a
candidate-level mechanism statistic. Baseline creates no correction labels and
must report zero continuation requests.

Interval artifacts store ID, parent ID, wall start/end, async status and
workload metadata. Concurrent request latency is reduced by interval union, not
summation. Parent exclusive time subtracts the union of children. `u_fixed` is
`unavailable` unless full-step coverage and parent/child validity are proven;
`other_wall` remains interval-union evidence instead of a potentially negative
naive residual.

The fixed-batch replay harness restores model, optimizer, Python/NumPy/Torch
RNG, and CUDA RNG before alternating diagnostics-off/on observations. It checks
loss, gradients, optimizer state, parameter update and RNG fingerprints. CUDA
events measure actor time, and identical-workload peak allocated/reserved memory
is used for the 2% memory gate. The 3% time gate is the maximum of fixed-replay
actor overhead and workload-normalized real-stage overhead; raw wall time from
independent stochastic launches is not an equivalence test.
The real profile launcher binds diagnostics `on` or `off` in its stage manifest
and writes the variants to separate namespaces. `evaluate_overhead.py` applies
the 3% time and 2% identical-workload memory gates.

## Standalone Ray topology

After separate approval to stop the old shared cluster, node A uses GCS 6397,
object/node ports 7111/7112 and workers 22000-22511. Node B uses GCS 6398,
object/node ports 7211/7212 and workers 23000-23511. Each advertises exactly
eight GPUs and `min(240, online_cpus-16)` CPUs under `ulimit -n 524288`.
Object-store capacity is the smaller node's floor-GiB value of
`min(20% MemTotal, 50% available /dev/shm, 128 GiB)` and must be at least 32
GiB. Ray, XDG, FlashInfer, Python, Torch-extension, log, metrics, profiler, and
output paths are node/stage-specific. Node B runs the one-shot controller
manually; there is no persistent HMAC service.

`preflight_node.py` rejects residual GPU compute processes and a wrong node IP;
profile, acceptance, and formal stages also verify W&B identity/connectivity.
`verify_local_ray.py` then requires exactly one live Ray node, eight GPUs, the
local hostname and exact GPU UUID inventory, and all declared fixed ports. The
one-shot controller owns cleanup only after its `ray start` succeeds, records
all teardown output, and `verify_teardown.py` requires both target-process and
port inventories to be empty.

Create one manifest per node only after the final code SHA is pushed. Use
`compare_node_manifests.py` to require equal code, recipe identity, model-file,
data, diagnostics, and frozen-artifact hashes while allowing different local
paths. `register_hard_prefix_panel.py` freezes at least 20 Base H=2048
length-finished prefixes using physical JSONL records and records both source
and panel SHA256. Acceptance automatically runs `validate_acceptance_log.py`
and `validate_checkpoint.py`; the latter loads all eight model/optimizer/extra
shards, requires RNG/scheduler/dataloader/counter state and an HF export, and
writes per-file SHA256.

## Cost and selection

Continuation control, input prefill, tail decode, long-reward row build, and
long-reward model forward are distinct. Unit costs are `u_request`,
`u_cont_input`, `u_tail_decode`, `u_long_row`, `u_long_token`, `u_normal`,
`u_actor`, and `u_candidate`. Raw and component-specific node-normalized values
are both retained. If the common Baseline calibration differs by more than 5%,
the fixed hard-prefix continuation panel and long-reward batch produce
component-specific crossover factors; arm placement does not change.

Moderate uses a shared 10% cap-hit scenario and cross-candidate p50 workloads;
Stress uses 30% and shared p90 workloads, with candidate batches capped at ten.
Unsafe or unstable candidates are removed, then Baseline must be within 10% of
the fastest Baseline. Selection minimizes the worse V1 Moderate/Stress regret.
A score tie within 5% is resolved by fixed-workload peak memory, warnings/retries,
smaller TP, then conservative offload/concurrency. Zero or fewer than 20 natural
requests triggers the frozen hard-prompt/cap-prefix panel; passing it preserves
system qualification but leaves natural mechanism coverage insufficient.

## Future S300 and entropy audit

Only a later explicit approval can start S300. Formal runs are one node/eight
GPUs, Step 0 validation, validation every 10, and full checkpoints every 50
through Step 300. An arm may resume only from a complete checkpoint in its own
formal directory, including model, optimizer, RNG, scheduler, dataloader and
counters.

Checkpoint entropy is benchmark-separated and uses rollout indices
`[0,8,16,24]` per question. The on-policy H8192 panel measures categorical
entropy on each checkpoint's visited states. The shared teacher-forced panel
freezes Base prompt/token histories, positions, seeds and SHA256 to isolate
conditional-distribution changes. Both report 256-token buckets, prefix/tail,
token-weighted/sequence-balanced, cap/non-cap and answer-transition cohorts.
Sampled-token surprisal is not a substitute for full-vocabulary categorical
entropy.
