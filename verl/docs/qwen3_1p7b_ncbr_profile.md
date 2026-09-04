# Qwen3-1.7B Baseline / NCBR-v1 profiling protocol

This protocol is pinned to `Qwen/Qwen3-1.7B-Base` revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, H=2048, L=8192,
B256/G768/M16/N8 and seed 42. A stage manifest must bind the clean code SHA,
the live immutable remote branch ref at that same SHA, diagnostics mode, node
identity, every node-local model file, Hugging Face
revision metadata, train/AIME2024/AIME2025 data hashes, and any frozen workload
or mechanism-panel artifact before a GPU process starts. AIME2024 and AIME2025
remain separate in every report.

The launcher is `examples/natural_continuation_boundary_return/run_qwen3_1p7b_profile_fsdp.sh`.
It accepts only the two arms, P0/P1/P2, and `calibration`, `gate0`, `profile`,
`mechanism_panel`, `fixed_replay`, `acceptance`, or `formal_s300`. Calibration itself is fixed to Baseline/P0.
`formal_s300` additionally fails closed unless the operator sets the explicit
authorization token after every preceding gate passes. The user has authorized
that automatic transition; profiling and five-step acceptance never write inside the
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
  verifies a complete loadable checkpoint. It then stops. Acceptance alone does
  not authorize S300, but after every calibration, Gate 0, profile, mechanism,
  replay, overhead, and acceptance gate passes, the already authorized S300
  transition is automatic. Recovery, Step 600, and another seed remain outside
  scope.

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

## Shared two-node Ray topology

This execution uses one user-approved shared 16-GPU cluster and never tears it
down between stages. Node A is the head at `10.8.191.127:6395`; both nodes use
object/node-manager ports 7011/7012, worker ports 21000-21511, and agent ports
52365-52368. Each advertises exactly eight GPUs, 240 CPUs, a 128 GiB object
store, and one unique resource (`ncbr_node_A` or `ncbr_node_B`) under
`ulimit -n 524288`. The 512-port range is mandatory: 21000-21099 was observed
to register only 99 of 240 prestarted workers and is rejected. Ray, XDG,
FlashInfer, Python, Torch-extension, log, metrics, profiler, and output paths
remain stage-specific. Node B is joined manually; there is no persistent HMAC
service. The launcher exports those writable cache roots locally and repeats
them in `ray_kwargs.ray_init.runtime_env.env_vars`, so a remote task runner
cannot fall back to read-only `/root/.cache` or import a different checkout.

`stage_local_assets.py` idempotently copies the model and all three parquet
files to `/tmp/qwen17-ncbr-assets-5904152e`, verifies every SHA256, and removes
all write bits. `verify_shared_ray.py` requires exactly two live labelled
nodes, 16 total and available GPUs, the exact hostname/IP/GPU inventories,
idle GPU memory, no compute process, both local asset sets, equal CPU/object
store capacity, and no worker-port registration error. The task runner, global
request load balancer, every AgentLoop/RewardLoop CPU actor, and every GPU
placement-group bundle request the selected `ncbr_node_A/B` resource. CPU
actors use hard node affinity and fail closed if the labelled node is
unavailable, so an arm cannot silently mix candidate or reward processing from
the peer node; an eight-GPU arm cannot land on the wrong node or span nodes.
The legacy standalone one-shot controller is not used for this shared-cluster
execution.

Create one manifest per node only after the final code SHA is pushed. Use
`compare_node_manifests.py` to require equal code, recipe identity, model-file,
data, diagnostics, and frozen-artifact hashes while allowing different local
paths. `register_hard_prefix_panel.py` freezes at least 20 Base H=2048
length-finished prefixes using physical JSONL records and records both source
and panel SHA256. Acceptance automatically runs `validate_acceptance_log.py`
and `validate_checkpoint.py`; the latter loads all eight model/optimizer/extra
shards, requires RNG/scheduler/dataloader/counter state and an HF export, and
writes per-file SHA256.

Calibration writes to node-specific directories, enables the exact boundary
identity recorder, exports the retained actor `DataProto`, runs a local fixed-batch
actor replay, and collects Base H=2048 cap rows. `compare_calibration_workloads.py`
strips node/process identity and requires prompt-token and candidate-batch inputs
to match exactly. Independent stochastic launches are not required to reproduce
response tokens, rewards, or retained ordering bitwise; those differences remain
recorded, and component costs are normalized by their actual token/row workloads. The cap source is
then frozen by `register_hard_prefix_panel.py`; Gate 0 and profile manifests
must carry its hashes and both calibration comparison receipts.
`sample_shared_gpus.py` is started only after both calibration logs enter the
training loop, samples all eight GPUs on each labelled node, and exits on an
explicit stop file without reserving or stopping a GPU.

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
`select_profile_candidate.py` consumes all six arm/candidate analyses plus
component node factors, applies the safety/stability and Baseline 10% gates,
then scores the worse normalized V1 Moderate/Stress regret. Candidate-level cap
rows are retained separately with transition, tail length, task delta, UID,
trajectory ID and `boundary_group_newly_locked`; that last field never enters
the actor batch.

For the final candidate, `fixed_replay` runs a real distributed actor
mini-batch. Each rank snapshots its local FSDP parameter shard, optimizer,
scaler and RNG state, performs an unmeasured allocator/optimizer warmup, then
alternates diagnostics off/on twice. `aggregate_fixed_actor_replay.py` requires
all eight ranks and exact loss, gradient, update, optimizer and RNG fingerprints.
`compare_profile_diagnostics.py` supplies only workload-normalized unit costs to
`evaluate_overhead.py`; stochastic raw step wall is excluded from the gate.

## S300 scheduling and entropy audit

After all gates pass, formal runs start without another approval. Each uses one
complete node/eight GPUs, Step 0 validation, validation every 10, and full
checkpoints every 50 through Step 300. If both nodes are idle, both arms start;
if only one is idle, Baseline starts there first and V1 starts when the other
complete node becomes idle. If neither is idle, resource monitoring is bounded
to 30 minutes and the same Baseline-first rule applies as soon as a complete
node is free. Partial GPU availability or unexplained memory/process occupancy
never qualifies. An arm may resume only from a complete checkpoint in its own
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

`generate_entropy_rollouts.py` produces 32 H8192 trajectories per unique AIME
question from Base or an HF checkpoint export, and
`annotate_entropy_transitions.py` joins checkpoint answers to Base by benchmark,
prompt and rollout index. `build_entropy_panel.py` then enforces rollout indices
`[0,8,16,24]` for every question and refuses pooled benchmark labels.
`audit_checkpoint_entropy.py`
streams the model with KV cache in bounded chunks, computes full-vocabulary
categorical entropy, and emits token-weighted plus sequence-balanced 256-token,
prefix/tail, cap and answer-transition cohorts separately for AIME2024 and
AIME2025. `estimate_s300.py` adds one Step-0 validation, 30 periodic validations
and six checkpoints to each of the Early/Moderate/Stress estimates.
`build_cumulative_axes.py` converts the formal per-step JSONL records into the
seven cumulative curve coordinates: optimizer step, candidate prompts, normal
decode tokens, continuation input tokens, continuation tail-decode tokens,
actor valid tokens, wall-clock, and GPU-hours. Baseline continuation coordinates
are checked to remain exactly zero.

The final `build_s300_gate.py` receipt is the only artifact that sets
`s300_authorized=true`; it requires both Gate 0, selection, both real fixed
replays, the diagnostics overhead gate, both acceptance validation/checkpoint
receipts, and the S300 estimate. `schedule_s300.py` then probes whole eight-GPU
Ray nodes, launches Baseline first, starts V1 only on the other complete idle
node, and bounds the no-capacity polling window to 30 minutes. It never stops
the shared Ray cluster and never authorizes Step 600 or a second seed.
