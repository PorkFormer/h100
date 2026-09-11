# Qwen3-1.7B eight-GPU implementation validation

Production source: `37a812d41ec7c1c5a070ee7741bed819f60aed55`; oracle: `d23da0e17830c296ae6e375793a7ccea98870bb3`; disabled baseline: `bfa08860fee9f4febaf7aa1041f0e7eb0ac09cd5`. Model: `/workspace/models/Qwen3-1.7B-Base`. Eight UUID-bound A100-SXM4-40GB GPUs, eight-rank FSDP and eight TP=1 rollout replicas. Full resolved configuration and exact commands are in each case directory. Model/tokenizer/config/data hashes and dependency versions: `manifest.json`, with live revalidation receipts. Frozen prompt manifest: `prompts.jsonl` / `prompts.parquet`.

## Verdict and limits

The completed gates below verify this model/configuration only. DAPO exhaustion leaves update coverage incomplete. Frozen cap-hit transitions were 24 smoke and 9 full, all 0→0. No natural nonzero task-score correction was observed; missing correctness transitions remain uncovered in real verifier evidence and are covered only by existing deterministic tests. Independent rollout tokens can differ; exact equivalence claims apply to frozen inputs/outputs. No training-effectiveness, capacity or performance improvement claim is made.

## Environment, replay and actor gates

- PASS: fresh per-device CUDA arithmetic, eight-rank NCCL sum 36 and UUID mapping; 190 CPU tests plus both byte-exact frozen characterizations (`cpu_release_retry/`).
- PASS: H128/L512 real capture (8 prompts / 32 rows / 24 cap hits), H2048/L8192 capture (32 prompts / 128 rows / 9 cap hits). Raw token IDs, finish metadata and unchanged math_dapo verifier outputs retained.
- PASS: frozen original runtime/adapter versus hook, off/shadow/replace request/mask/score/order equality (`replay_h128_v3/`, `replay_h2048_v3/`). Smoke independent repeats differ; full repeats happened to match in this sample.
- PASS: actual eight-rank FSDP registered vanilla/GSPO losses and backward. H128 uses 32 rows; H2048 uses first 2 prompts / 8 rows. Repeated full loss inputs and gradient shards, and cross-mode results, compare exactly (`actor_h128/`, `actor_h2048/`, `actor_cross_mode.json`).
- PASS: GPU GSPO masked-tail isolation; original GRPO recomputation with controlled score change changes gradient. The score change is synthetic, separately labeled (`tail_isolation.json`).
- PASS: reference forward at KL coefficient zero; eight workers have exact input IDs/positions/masks and prefix logprobs of shape [8,2048], initial old/ref equality (`reference_alignment.json`).

## Trainer and diagnostic cases

Optimizer steps count actual AdamW calls across all eight ranks. Changed shards count receipts with different before/after checksums. Version columns distinguish publication completion and actual normal rollout metadata. Two-step cases publish version 1 before step two and version 2 at completion. Serving the nonzero step-two update (version 2) is UNCOVERED in standard cases; only versions 0/1 were actually used by normal rollouts. DAPO replace separately covers actual serving after a nonzero update.

| Case | Status | Steps | Changed shard receipts | Published | Served normal | Seconds |
|---|---|---:|---:|---|---|---:|
| train_dapo_vanilla_off_v6 | UNCOVERED_DAPO_EXHAUSTION | 0 | 0 | [0] | [0] | 264.2 |
| train_dapo_vanilla_replace_v6 | UNCOVERED_DAPO_EXHAUSTION | 1 | 8 | [0, 1] | [0, 1] | 612.1 |
| train_dapo_vanilla_shadow_v6 | UNCOVERED_DAPO_EXHAUSTION | 0 | 0 | [0] | [0] | 278.0 |
| train_standard_gspo_off_v6 | PASS | 2 | 8 | [0, 1, 2] | [0, 1] | 209.5 |
| train_standard_gspo_replace_fault_duplicate_v6 | PASS | 0 | 0 | [0] | [0] | 161.4 |
| train_standard_gspo_replace_fault_missing_v6 | PASS | 0 | 0 | [0] | [0] | 164.3 |
| train_standard_gspo_replace_fault_release_v6 | PASS | 0 | 0 | [0] | [0] | 157.9 |
| train_standard_gspo_replace_fault_verifier_error_v6 | PASS | 0 | 0 | [0] | [0] | 162.7 |
| train_standard_gspo_replace_fault_verifier_timeout_v6 | PASS | 0 | 0 | [0] | [0] | 163.9 |
| train_standard_gspo_replace_fault_version_v6 | PASS | 0 | 0 | [0] | [0] | 161.2 |
| train_standard_gspo_replace_live_oracle_full_v6 | PASS | 0 | 0 | [0] | [0] | 358.3 |
| train_standard_gspo_replace_live_oracle_smoke_v6 | PASS | 0 | 0 | [0] | [0] | 231.2 |
| train_standard_gspo_replace_ref_v6 | PASS | 1 | 0 | [0, 1] | [0] | 181.1 |
| train_standard_gspo_replace_v6 | PASS | 2 | 8 | [0, 1, 2] | [0, 1] | 235.9 |
| train_standard_gspo_shadow_v6 | PASS | 2 | 8 | [0, 1, 2] | [0, 1] | 237.4 |
| train_standard_vanilla_off_v6 | PASS | 2 | 8 | [0, 1, 2] | [0, 1] | 210.3 |
| train_standard_vanilla_replace_v6 | PASS | 2 | 8 | [0, 1, 2] | [0, 1] | 234.2 |
| train_standard_vanilla_shadow_v6 | PASS | 2 | 8 | [0, 1, 2] | [0, 1] | 238.2 |

Standard main cases have two optimizer calls; the first has zero gradients, and the second changes all eight shards. DAPO off/shadow exhaust four candidate rounds without update; DAPO replace changes all eight shards once, serves version 1, then exhausts the next candidate budget. Data, reward and filtering were not adjusted.

## Live oracle and fault barriers

- PASS: live smoke, 27 requests, exact requests/masks and frozen adapter scores, 12 independent tail differences, zero optimizer updates. Controlled pre-update exit is expected.
- PASS: live full, 4 requests, exact requests/masks and frozen adapter scores, 1 independent tail differences, zero optimizer updates. Controlled pre-update exit is expected.
- PASS: injected missing, 8 injection receipts, 0 optimizer receipts, published versions [0].
- PASS: injected duplicate, 8 injection receipts, 0 optimizer receipts, published versions [0].
- PASS: injected version, 8 injection receipts, 0 optimizer receipts, published versions [0].
- PASS: injected verifier_error, 1 injection receipts, 0 optimizer receipts, published versions [0].
- PASS: injected verifier_timeout, 1 injection receipts, 0 optimizer receipts, published versions [0].
- PASS: injected release, 5 injection receipts, 0 optimizer receipts, published versions [0].

Faults are controlled callback/service-boundary injections, including a synthetic TimeoutError, not induced wall-clock network outages. Unconfirmed release requires no later sleep attempt. Expected nonzero exits are PASS only when the suite verifies injection/completion receipts and update barriers; timeout never passes.

## Evidence and resources

`case_analysis.json` records exact actor short-prefix identity, finite loss/gradient metrics, actual UUIDs, continuation/release counts and reward chunk padding. Standard enabled cases exercise real 3-row reward chunks padded to 4 across two workers. Engine/worker import receipts exist for every main case; expanded server/reward module and remote release/drain acknowledgement instrumentation was added for later live/fault diagnostics. Earlier main cases retain runtime cleanup intervals rather than separate remote acknowledgement files. Worker audit files retain raw reward and actor batches, gradients, weight hashes, publication receipts, import paths, release/drain evidence and profiling intervals. Raw scores remain saved separately; corrections are restricted to the last valid H token; no continuation tail enters the actor.

- H128 direct capture: 57.1 s, sampled NVML peak per GPU [14335, 14357, 14335, 14337, 14357, 14335, 14375, 14335] MiB. Actor replay maximum CUDA allocated: 11.707 GiB.
- H2048 direct capture: 266.7 s, sampled NVML peak per GPU [14451, 14457, 14453, 14611, 14335, 14335, 14335, 14565] MiB. Actor replay maximum CUDA allocated: 15.540 GiB.

Trainer per-case wall time is listed above; CUDA allocation/reservation metrics are retained in `case_analysis.json`. NVML samples, CUDA allocation counters and profiling intervals measure different scopes and are not equated. Audit I/O overhead is included; timing equality is not asserted.

## Isolation, corrections and reproduction

No production algorithm/source fix was needed. Private runner fixes handle vLLM UUID parsing with a lazy import shim, Ray device assignment, private short IPC paths, CPU placement capacity and typed seed configuration. Earlier failed attempts and original logs remain in evidence. The final release-fault launch was first blocked by a transient GPU0 process-occupancy snapshot before trainer startup; its directory is retained with `_gate_attempt1`. The launcher now requires two idle snapshots with an empty compute-process list, and the release case was retried only after a fresh gate. No golden/detector/loss mathematics or shared dependencies changed. No formal checkpoint or benchmark evaluation ran.

Reproduction commands and fixed parameters: `../README.md`. Each trainer directory includes `config.yaml`, `process.json`, gate and runtime logs. Final cleanup and provenance receipts: `final_state.json`, `final_manifest.json`. Final inspection found no owned processes and all eight GPUs at 0 MiB / 0% utilization; no forced process termination was needed. Validation code is committed separately from raw evidence/cache files.
